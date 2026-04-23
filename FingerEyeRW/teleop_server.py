import zmq
import time
import json
import hydra
import threading
import numpy as np
from pathlib import Path
from termcolor import cprint
from omegaconf import OmegaConf
from xarm.wrapper import XArmAPI
from leap_hand_utils.leapnode import LeapNode
from recorder_utils.recorder_storage import disable_and_refill

class TeleopServer:
    def __init__(self, config):
        """Initialize the robot server with ZMQ sockets."""
        self.config = config
        self.ctx = zmq.Context.instance()
        self.rep = self.ctx.socket(zmq.REP)
        self.rep.bind("tcp://*:5557")
        self.rep.setsockopt(zmq.RCVTIMEO, 1000)
        self.zmq_thread = None
        self.zmq_running = False

        self.hand_server = LeapNode(enable_hand=False)
        self.init_xarm()

    def zmq_request_handler(self):
        while self.zmq_running:
            try:
                request = self.rep.recv_string()
                request_data = json.loads(request)
                response = self.handle_request(request_data)
                self.rep.send_string(json.dumps(response))
            except zmq.Again:
                continue
            except Exception as e:
                print(f"Error handling request: {e}")
                self.rep.send_string(json.dumps({"status": "error", "message": str(e)}))

    def start_zmq_handler(self):
        self.zmq_running = True
        self.zmq_thread = threading.Thread(target=self.zmq_request_handler)
        self.zmq_thread.daemon = True  
        self.zmq_thread.start()
        print("✅ ZMQ Handler thread started")

    def stop_zmq_handler(self):
        self.zmq_running = False
        if self.zmq_thread and self.zmq_thread.is_alive():
            self.zmq_thread.join(timeout=1.0)
            print("✅ ZMQ Handler thread stopped")

    def init_xarm(self):
    
        self.xarm_api = XArmAPI(self.config.server_xarm_api)
        self.xarm_connected = True
        print("✅ Connect to xArm7!")

        self.xarm_api.set_simulation_robot(False)
        self.xarm_api.clean_error()
        self.xarm_api.clean_warn()
        self.xarm_api.motion_enable(True)
        if not self.config.use_keyboard_arm_tele:
            self.xarm_goto_home_pose()

    def xarm_goto_home_pose(self):
        self.xarm_api.set_mode(0)
        time.sleep(0.1)
        self.xarm_api.set_state(state=0)
        time.sleep(0.1)
        self.xarm_init_joint_values_degree = self.config.xarm_init_joint_values_degree
        self.xarm_api.set_servo_angle(
            angle=self.xarm_init_joint_values_degree,
            speed=15,  
            acceleration=5,  
            wait=True,
            timeout=5,
            is_radian=False,
        )  
        print("✅ XArm to initial position.")
        time.sleep(0.1)

        if self.config.use_kin_arm_tele:
            code = self.xarm_api.set_mode(2)
            time.sleep(0.1)
            self.xarm_api.set_state(0)

    def handle_request(self, request):  
        """Handle incoming ZMQ requests.
        
        Args:
            request (dict): The parsed JSON request
            
        Returns:
            dict: Response to send back to client
        """
        command = request.get("command")
        
        if command == "get_leap_joint_values":
            self.hand_joints = np.array(self.hand_server.read_pos())
            self.hand_joints = disable_and_refill(self.hand_joints, self.config.tele_leap_disabled_idx, self.config.leap_init_joint_values_radian, dim_to_slice=-1, refill_values=np.pi)
            cprint(self.config.tele_leap_disabled_idx, "yellow")
            print(self.hand_joints)
            
            return {"angles": self.hand_joints.tolist()}
        
        elif command == "get_all_joint_values":
            code, angles = self.xarm_api.get_servo_angle()
            self.xarm_joints = np.array(angles)
            self.xarm_joints = np.deg2rad(self.xarm_joints)
            self.hand_joints = np.array(self.hand_server.read_pos())
            self.hand_joints = disable_and_refill(self.hand_joints, self.config.tele_leap_disabled_idx,self.config.leap_init_joint_values_radian, dim_to_slice=-1, refill_values=np.pi)
            print(self.hand_joints)
            return {"angles": np.concatenate([self.xarm_joints, self.hand_joints]).tolist()}
        
        elif command == "init":
            if not self.config.use_keyboard_arm_tele:
                self.xarm_goto_home_pose()
            return {"status": "success", "message": "Robot initialized"}
        
        else:
            return {"status": "error", "message": "Unknown command"}

    def run(self):
        self.start_zmq_handler()
        while True:
            time.sleep(0.01)
  
    def __del__(self):
        self.stop_zmq_handler()

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.joinpath("configs")),
    config_name="coin_standing.yaml",
)
def main(config):
    cprint(OmegaConf.to_yaml(config), "grey")
    server = TeleopServer(config)

    try:
        server.run()
    except KeyboardInterrupt:
        print("\n🛑 Stop Teleop Server")
    finally:
        server.stop_zmq_handler()

if __name__ == "__main__":
    main()

