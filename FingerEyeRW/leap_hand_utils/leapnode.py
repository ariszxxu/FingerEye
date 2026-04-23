import numpy as np 
import time 
import threading
from .dynamixel_client import DynamixelClient

class LeapNode:
    def __init__(self, kP=600, kD=200, kp_side=0.75, kd_side=0.75, init_joint_values_radian=np.ones(16) * np.pi, enable_hand=True):
        self.kP = kP
        self.kD = kD
        self.kp_side = kp_side
        self.kd_side = kd_side
        self.enable_hand = enable_hand

        self.kI = 0
        self.curr_lim = 700
        self.prev_pos = self.curr_pos = init_joint_values_radian + np.pi
        self.motors = list(range(16))
        self.dxl_client = None
        self.connected = False

        self.last_read_time = 0
        self.read_interval = 0.0  
        self.cached_position = np.ones(16) * np.pi
        self.read_lock = threading.Lock() 

        self.current_port = None
        self.available_ports = ['/dev/ttyUSB0', '/dev/ttyUSB1','/dev/ttyUSB2','/dev/ttyUSB3']
        self.connection_retry_count = 0
        self.max_retry_attempts = 3
        self.last_successful_port = None
        self.connection_lost_time = None
        self.reconnection_interval = 2.0  

        self._attempt_connection()

    def _attempt_connection(self):
        try:
            self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
            self.connected = True
            self.current_port = '/dev/ttyUSB0'
            self.last_successful_port = '/dev/ttyUSB0'
            self.connection_retry_count = 0
            self.connection_lost_time = None
            print(f"✅ LEAP Hand Connect to (/dev/ttyUSB0)")
            self._initialize_parameters()
            return True
        except Exception:
            try:
                self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
                self.connected = True
                self.current_port = '/dev/ttyUSB1'
                self.last_successful_port = '/dev/ttyUSB1'
                self.connection_retry_count = 0
                self.connection_lost_time = None
                print(f"✅ LEAP Hand Connect to (/dev/ttyUSB1)")
                self._initialize_parameters()
                return True
            except Exception:
                try:
                    self.dxl_client = DynamixelClient(self.motors, '/dev/ttyUSB2', 4000000)
                    self.dxl_client.connect()
                    self.connected = True
                    self.current_port = '/dev/ttyUSB2'
                    self.last_successful_port = '/dev/ttyUSB2'
                    self.connection_retry_count = 0
                    self.connection_lost_time = None
                    print(f"✅ LEAP Hand Connect to (/dev/ttyUSB2)")
                    self._initialize_parameters()
                    return True
                except Exception:
                    print("❌ Fail to connect to LEAP Hand")
                    self.dxl_client = None
                    self.connected = False
                    self.current_port = None
                    return False

    def _initialize_parameters(self):
        if not self.connected or not self.dxl_client:
            return

        self.dxl_client.set_torque_enabled(self.motors, self.enable_hand)
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kP, 84, 2)  # Pgain stiffness
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kP * self.kp_side), 84, 2)  # Pgain stiffness for side to side should be a bit less

        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kI, 82, 2)  # Igain
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.kD, 80, 2)  # Dgain damping
        self.dxl_client.sync_write([0, 4, 8], np.ones(3) * (self.kD * self.kd_side), 80, 2)  # Dgain damping for side to side should be a bit less
        
        # Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(self.motors, np.ones(len(self.motors)) * self.curr_lim, 102, 2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    def _handle_connection_error(self):
        if self.connected:
            print(f"⚠️ LEAP Hand Connection Lost (Port: {self.current_port})")
            self.connected = False
            self.connection_lost_time = time.time()

            # Clean up the current connection.
            if self.dxl_client:
                try:
                    self.dxl_client.disconnect()
                except:
                    pass
                self.dxl_client = None

    def set_leap(self, pose):
        '''
        input: viser_leap_joints
        '''
        if not self.connected or not self.dxl_client:
            self._check_and_reconnect()
            return

        with self.read_lock:
            try:
                self.prev_pos = self.curr_pos
                self.curr_pos = np.array(pose) + np.pi
                self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
            except OSError as e:
                if "Port is in use" in str(e):
                    print(f"Port is in use, try to reconnect...")
                    self._handle_connection_error()
                else:
                    print(f"⚠️ LEAP Hand Set Position Error: {e}")
                    self._handle_connection_error()
            except Exception as e:
                print(f"⚠️ LEAP Hand Set Position Error: {e}")
                self._handle_connection_error()

    def read_pos(self):
        """
        out_put: viser_leap_joints
        """
        if not self.connected or not self.dxl_client:
            self._check_and_reconnect()
            return self.cached_position - np.pi

        current_time = time.time()

        if current_time - self.last_read_time < self.read_interval:
            return self.cached_position - np.pi

        with self.read_lock:
            try:
                position = self.dxl_client.read_pos()
                if position is not None:
                    self.cached_position = np.array(position)
                    self.last_read_time = current_time
                    return self.cached_position - np.pi
                else:
                    return self.cached_position - np.pi
            except OSError as e:
                if "Port is in use" in str(e):
                    print(f"Port is in use, try to reconnect...")
                    self._handle_connection_error()
                    if self._attempt_connection():
                        print("✅ LEAP Hand Reconnect Success!")
                        try:
                            position = self.dxl_client.read_pos()
                            if position is not None:
                                self.cached_position = np.array(position)
                                self.last_read_time = current_time
                                return self.cached_position - np.pi
                        except:
                            pass
                else:
                    print(f"⚠️ LEAP Hand Read Position Error: {e}")
                    self._handle_connection_error()

                return self.cached_position - np.pi
            except Exception as e:
                if "TxRxResult" in str(e) or "Port" in str(e):
                    print(f"⚠️ LEAP Hand Communication Error: {e}")
                    self._handle_connection_error()
                    if self._attempt_connection():
                        print("✅ LEAP Hand Reconnect Success!")
                        try:
                            position = self.dxl_client.read_pos()
                            if position is not None:
                                self.cached_position = np.array(position)
                                self.last_read_time = current_time
                                return self.cached_position - np.pi
                        except:
                            pass
                else:
                    print(f"⚠️ LEAP Hand Read Position Error: {e}")

                return self.cached_position - np.pi

    def get_connection_status(self):
        current_time = time.time()

        if self.connected:
            return f"🟢 Connected to ({self.current_port})"
        elif self.connection_lost_time:
            time_since_lost = current_time - self.connection_lost_time
            return f"🔴 Lose Connection {time_since_lost:.1f}s (Retry: {self.connection_retry_count}/{self.max_retry_attempts})"
        else:
            return "🔴 Not Connected"

    def safe_disconnect(self):

        print("🔄 Safe Disconnect LEAP Hand...")

        with self.read_lock:
            try:
                print("🔄 Disconnect LEAP Hand...")
                self.dxl_client.disconnect()

                self.connected = False
                self.dxl_client = None
                self.current_port = None
                print("✅ LEAP Hand Safe Disconnect Success!")

            except Exception as e:
                print(f"⚠️ LEAP Hand Safe Disconnect Error: {e}")
            finally:
                self.connected = False
                self.dxl_client = None
                self.current_port = None

