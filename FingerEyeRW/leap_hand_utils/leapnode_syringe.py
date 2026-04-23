import numpy as np
import time
import threading
from .dynamixel_client import DynamixelClient


class LeapNode:
    def __init__(
        self,
        kP=600,
        kD=200,
        kp_side=0.75,
        kd_side=0.75,
        kp_T_tip=1,
        kd_T_tip=1,
        init_joint_values_radian=np.ones(16) * np.pi,
        enable_hand=True,
        record_frequency=10,   # Added parameter: match control frequency.
    ):
        self.kP = kP
        self.kD = kD
        self.kp_side = kp_side  
        self.kd_side = kd_side
        self.kp_T_tip=kp_T_tip
        self.kd_T_tip=kd_T_tip
        self.enable_hand = enable_hand
        self.record_frequency = record_frequency
        self.record_interval = 1.0 / record_frequency

        self.kI = 0
        self.curr_lim = 550
        self.prev_pos = self.curr_pos = init_joint_values_radian.copy() + np.pi
        self.target_pos = self.curr_pos.copy()
        self.motors = list(range(16))
        self.dxl_client = None
        self.connected = False

        self.read_lock = threading.Lock()
        self.cached_position = np.ones(16) * np.pi
        self.last_read_time = 0
        self.read_interval = 0.0

        self.available_ports = [
            "/dev/ttyUSB0",
            "/dev/ttyUSB1",
            "/dev/ttyUSB2",
            "/dev/ttyUSB3",
        ]
        self.connection_retry_count = 0
        self.max_retry_attempts = 3
        self.current_port = None
        self.last_successful_port = None
        self.connection_lost_time = None
        self.reconnection_interval = 2.0

        self._attempt_connection()

        self._stop_thread = False
        self.control_thread = threading.Thread(target=self._smooth_worker, daemon=True)
        self.control_thread.start()

    def _attempt_connection(self):
        for port in self.available_ports:
            try:
                self.dxl_client = DynamixelClient(self.motors, port, 4000000)
                self.dxl_client.connect()
                self.connected = True
                self.current_port = self.last_successful_port = port
                print(f"✅ LEAP Hand Connected ({port})")
                self._initialize_parameters()
                return True
            except Exception:
                continue
        print("❌ Fail to connect to LEAP Hand")
        self.dxl_client = None
        self.connected = False
        return False

    def _initialize_parameters(self):
        if not self.connected or not self.dxl_client:
            return
        self.dxl_client.set_torque_enabled(self.motors, self.enable_hand)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * self.kp_side), 84, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * self.kd_side), 80, 2)
        self.dxl_client.sync_write([14, 15], np.ones(2) * (self.kP * self.kp_T_tip), 84, 2)
        self.dxl_client.sync_write([14, 15], np.ones(2) * (self.kD * self.kd_T_tip), 80, 2)


        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def _handle_connection_error(self):
        if self.connected:
            print(f"⚠️ LEAP Hand Connection Lost (Port: {self.current_port})")
            self.connected = False
            self.connection_lost_time = time.time()
            if self.dxl_client:
                try:
                    self.dxl_client.disconnect()
                except:
                    pass
                self.dxl_client = None

    def set_leap(self, pose):
        if not self.connected or not self.dxl_client:
            self._check_and_reconnect()
            return
        with self.read_lock:
            self.target_pos = np.array(pose, dtype=float) + np.pi



    def _smooth_worker(self):
        rate_hz = 100
        dt = 1.0 / rate_hz
        total_time = 0.2      
        n_steps = 5         
        segment_time = total_time / n_steps
        steps_per_segment = int(segment_time / dt)
        prev_target = self.target_pos.copy()

        while not self._stop_thread:
            try:
                if not (self.connected and self.dxl_client and self.target_pos is not None):
                    time.sleep(dt)
                    continue

                if not np.allclose(prev_target, self.target_pos, atol=1e-6):
                    prev_target = self.target_pos.copy()
                    start_pos = self.curr_pos.copy()
                    target_pos = self.target_pos.copy()

                    sub_targets = np.linspace(start_pos, target_pos, n_steps + 1)[1:]

                    for sub_idx, sub_tgt in enumerate(sub_targets):
                        if not np.allclose(self.target_pos, prev_target, atol=1e-6):
                            break

                        for _ in range(steps_per_segment):
                            self.curr_pos = sub_tgt
                            with self.read_lock:
                                self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
                            time.sleep(dt)

                else:
                    time.sleep(dt)

            except Exception as e:
                print(f"⚠️ Smooth Control Thread Error: {e}")
                self._handle_connection_error()
                time.sleep(1.0)

    def read_pos(self):
        if not self.connected or not self.dxl_client:
            self._check_and_reconnect()
            return self.cached_position - np.pi
        with self.read_lock:
            try:
                pos = self.dxl_client.read_pos()
                if pos is not None:
                    self.cached_position = np.array(pos)
                    self.last_read_time = time.time()
            except Exception as e:
                print(f"⚠️ Read Position Error: {e}")
                self._handle_connection_error()
        return self.cached_position - np.pi

    def safe_disconnect(self):
        print("🔄 safely disconnect LEAP Hand...")
        self._stop_thread = True
        if hasattr(self, "control_thread"):
            self.control_thread.join(timeout=1.0)
        with self.read_lock:
            try:
                if self.dxl_client:
                    self.dxl_client.disconnect()
                print("✅ LEAP Hand Safely Disconnected")
            except Exception as e:
                print(f"⚠️ Safe Disconnect Error: {e}")
            finally:
                self.connected = False
                self.dxl_client = None
                self.current_port = None