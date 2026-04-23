import cv2
import zmq
import json
import time 
import math
import torch
import hydra
import viser
import os
import importlib
import numpy as np
from pathlib import Path
from copy import deepcopy
from termcolor import cprint
from collections import deque
from omegaconf import OmegaConf
from xarm.wrapper import XArmAPI
from viser.extras import ViserUrdf
# from leap_hand_utils.leapnode import LeapNode
from recorder_utils.recorder_rs import RealSenseManager
from recorder_utils.recorder_av_cam import AVCameraManager
from recorder_utils.recorder_keyboard import KeyboardActionManager
from recorder_utils.recorder_storage import RecorderStorage, slice_with_list, refill_full_list_with_slice_indices, precise_wait, disable_and_refill
# from recorder_utils.recorder_aux_cam import AUXCameraManager
import sys 
sys.path.append("./recorder_utils")

from recorder_utils.recorder_img_utils import *
from recorder_utils.recorder_tag_detector import AprilTagDetector


server = None
viser_urdf = None

def extract_RT_stack(tag_dict):
    """
    tag_dict: dict with keys like 'I', 'T', each value is (4,4) homogeneous matrix
    return: np.ndarray of shape (2, 3, 4)
             where [:, :3, :3] = rotation, [:, :3, 3:] = translation
    """
    # Keep a deterministic key order.
    keys = tag_dict.keys()

    # Extract each tag's 3x4 matrix [R|t].
    mats = []
    for k in keys:
        T = np.asarray(tag_dict[k])
        assert T.shape == (4, 4), f"{k} must be 4x4"
        R = T[:3, :3]
        t = T[:3, 3:4]   # Keep translation as a (3,1) column vector.
        Rt = np.concatenate([R, t], axis=1)  # (3,4)
        mats.append(Rt)

    return np.stack(mats, axis=0)

def compute_delta_transform_like_your_version(T_ref, T_cur):
    R_ref = T_ref[:3, :3]
    t_ref = T_ref[:3, 3]

    R_cur = T_cur[:3, :3]
    t_cur = T_cur[:3, 3]

    # rotation
    R_rel = R_ref.T @ R_cur

    # translation: rotate into ref frame
    t_rel = R_ref.T @ (t_cur - t_ref)

    return np.concatenate([R_rel, t_rel.reshape(3, 1)], axis=1)

def concat_and_pad_obs(
    obs_deque: deque,
    To: int,
):
    """
    Args:
        obs_deque: deque of dict[str, Tensor]
            each tensor shape: (B, ...) or (B, 1, ...)
        To: required timestep length

    Returns:
        dict[str, Tensor] with shape (B, To, ...)
    """
    assert len(obs_deque) > 0, "obs_deque is empty"

    # use first element as template
    keys = obs_deque[0].keys()
    out = {}

    for k in keys:
        # collect tensors, ensure time dim exists
        seq = []
        for obs in obs_deque:
            v = obs[k]
            if v.dim() == 2 or v.dim() >= 3 and v.shape[1] != 1:
                # (B, ...) → (B, 1, ...)
                v = v.unsqueeze(1)
            seq.append(v)

        x = torch.cat(seq, dim=1)  # (B, T_cur, ...)

        T_cur = x.shape[1]
        if T_cur < To:
            pad = x[:, :1].repeat(1, To - T_cur, *([1] * (x.dim() - 2)))
            x = torch.cat([pad, x], dim=1)
        elif T_cur > To:
            x = x[:, -To:]

        out[k] = x

    return out

class TagPoseHistory:
    def __init__(self, window=4):
        self.window = window
        self.buffer = deque(maxlen=window + 1)
        self.base_pose = None  # Reference 4x4 matrices (e.g. tags 'I' and 'T').

    def get_new_tag_pose(self, tag_pose, first_frame=False):
        """
        Receive a new tag_pose.
        - If this is the first frame (or a reset), update base_pose.
        - Otherwise compute the delta (2,3,4) from base_pose.
        Append the result to the history buffer.
        """
        tag_pose = extract_RT_stack(tag_pose)  # shape (2,3,4)
        if first_frame or self.base_pose is None:
            self.base_pose = tag_pose
            self.buffer.clear()
        # Compute the current-frame delta_IT relative to base_pose.
        delta = np.array([compute_delta_transform_like_your_version(self.base_pose[i], tag_pose[i])
                          for i in range(len(tag_pose))])  # shape (2,3,4)
        self.buffer.append(delta)  # shape (2,3,4)

    def get_padded_history(self):
        """
        Return shape (2, window+1, 3, 4).
        Pad history using the earliest available frame when needed.
        """
        if len(self.buffer) == 0:
            raise ValueError("No tag_pose data has been received yet.")

        n = len(self.buffer)
        n_tags = self.buffer[0].shape[0]  # Expected 2 (I and T).
        out = np.zeros((n_tags, self.window + 1, 3, 4))
        last_idx = 0
        indices = []

        for k in range(self.window + 1):
            target_idx = n - (self.window + 1 - k)
            if target_idx < 0:
                target_idx = last_idx
            else:
                last_idx = target_idx
            indices.append(target_idx)

        indices = np.clip(indices, 0, n - 1)
        for i in range(n_tags):  # Gather history per tag channel (I / T).
            out[i] = np.array([self.buffer[j][i] for j in indices])
        return out.reshape(n_tags, -1)  # (n_tags, 60)

class ConTacTeleop:
    def __init__(self, config, LeapNode=None):
        self.config = config
        self.LeapNode = LeapNode
        self.mode = self.config.mode
        cprint(f"Mode: {self.mode}", color="cyan", attrs=["bold"])
        assert self.mode in ["tele_test", "record", "policy"], f"Invalid mode: {self.mode}"
        self.moving_to_max,self.moving_to_min = False,False
        self.record_frequency = int(self.config.record_frequency) 
        self.record_interval = 1.0 / self.record_frequency
        self.action_interval = self.record_interval * 0.03  
        self.obs_interval = self.record_interval - self.action_interval
        # self.saved_aux = False
        # self.saved_aux_test = 0
        self.sampling_enabled =True
        self.use_action_from_teleop = self.mode in ["tele_test", "record"]
        self.first_time = True
        self.target_xarm_onehot = None
        if "xarm_target_joint_values_degree_list" in self.config:
            self.target_xarm_onehot = np.zeros(len(self.config.xarm_target_joint_values_degree_list),)
        self.use_cam_obs =  self.mode in ["record", "policy"]
        self.camera_names = []
        self.rs_camera_names = []
        self.tag_pose_history = TagPoseHistory(window=4)
        if self.use_cam_obs:
            self.init_cameras()

        self.recorder_storage = RecorderStorage()

        self.init_xarm()
        if self.config.use_mixed_tele:
            self.init_server_xarm()
        self.init_leap_hand()
        self.init_viser_server()

        if self.use_action_from_teleop:
            true_count = (
                self.config.use_keyboard_arm_tele + 
                self.config.use_kin_arm_tele + 
                self.config.use_mixed_tele
            )
            assert true_count == 1, f"Exactly one teleop control mode must be enabled, but got {true_count}."

            # both teleop modes need to get joint values from zmq
            self.init_zmq()
            # only keyboard teleop needs keyboard manager
            if self.config.use_keyboard_arm_tele or self.config.use_mixed_tele:
                self.keyboard_tele = KeyboardActionManager()
        else: 
            # policy mode 
            self.load_policy()

    def load_policy(self):
        from contac.workspaces.workspace import TrainWorkspace
        assert Path(self.config.eval_ckpt_path).exists(), f"❌ Cannot find eval checkpoint: {self.config.eval_ckpt_path}"
        # if not given self.config.eval_ckpt_config_path 
        if not hasattr(self.config, 'eval_ckpt_config_path') or self.config.eval_ckpt_config_path is None:
            # try to find the config file in the same dir as the checkpoint
            self.config.eval_ckpt_config_path = Path(self.config.eval_ckpt_path).parent.parent / ".hydra" / "config.yaml"

        assert Path(self.config.eval_ckpt_config_path).exists(), f"❌ Cannot find eval config: {self.config.eval_ckpt_config_path}"
        workspace_config = OmegaConf.load(self.config.eval_ckpt_config_path)
        self.workspace_config = workspace_config
        workspace_config.eval_ckpt_path = self.config.eval_ckpt_path
        cprint(OmegaConf.to_yaml(workspace_config), "grey")
        workspace = TrainWorkspace(workspace_config)
        self.policy = workspace.run_eval()
        self.actions_to_rollout = deque(maxlen=workspace_config.n_action_steps)
        self.obs_to_takein = deque(maxlen=workspace_config.n_obs_steps)
        self.n_obs_steps = workspace_config.n_obs_steps
        # get the action form 
        replay_buffer_keys = workspace_config.task.dataset.replay_buffer_keys
        action_key = None 
        if "actions/original_actions" in replay_buffer_keys:
            action_key = "actions/original_actions"
            action_form = "absolute_joint_values"
        else:
            action_form = None

        assert action_form is not None, f"❌ Cannot find action form in replay_buffer_keys: {replay_buffer_keys}"
        self.action_key = action_key
        self.action_form = action_form

    def init_cameras(self):
        self.enabled_stereo_camera_names = list(self.config.enabled_stereo_camera_names)
        self.camera_names = list(self.config.camera_names)
        self.enabled_camera_to_port = {name: port for name, port in self.config.camera_to_port.items() if name in self.config.enabled_stereo_camera_names}
        self.enabled_camera_left_right_order = {name: order for name, order in self.config.camera_left_right_order.items() if name in self.config.enabled_stereo_camera_names}
        self.camera_manager = AVCameraManager(self.enabled_camera_to_port, camera_left_right_order=self.enabled_camera_left_right_order, default_options=self.config.default_camera_opts)
        self.camera_manager.open_all_cameras()
        self.tag_detector = AprilTagDetector(self.config)
        self.rs_camera_names = list(self.config.rs_camera_names)
        self.realsense_manager = RealSenseManager(desired_fps=self.config.rs_fps,)

    def init_xarm(self):
        try:
            self.xarm_api = XArmAPI(self.config.xarm_api)

            if self.xarm_api.connected:
                self.xarm_connected = True
                print("✅ Connect to xArm!")
                self.xarm_api.set_simulation_robot(False)
                self.xarm_api.clean_error()
                self.xarm_api.clean_warn()
                self.xarm_api.motion_enable(True)
                time.sleep(0.1)
                self.xarm_api.set_mode(0) 
                time.sleep(0.1)
                self.xarm_api.set_state(state=0)
                time.sleep(0.1)
                self.xarm_api.set_servo_angle(
                    angle=np.array(self.config.xarm_init_joint_values_degree),
                    speed=15, 
                    wait=True,
                    is_radian=False,
                )  
                print("🤖 xArm to initial configuration!")
                if self.config.use_keyboard_arm_tele:
                    self.xarm_api.set_mode(5) 
                elif self.config.use_kin_arm_tele or self.config.use_mixed_tele:
                    self.xarm_api.set_mode(6)
                self.xarm_api.set_state(state=0)
                time.sleep(0.1)
            else:
                print("❌ Fail to connect to xArm!")
                self.xarm_connected = False

        except Exception as e:
            print(f"❌ Error in init_xarm: {e}")
            self.xarm_connected = False
    
    def init_server_xarm(self):
        try:
            self.server_xarm_api = XArmAPI(self.config.server_xarm_api)
            
            if self.server_xarm_api.connected:
                self.server_xarm_connected = True
                print("✅ Connect to server xArm!")
                self.server_xarm_api.set_simulation_robot(False)
                self.server_xarm_api.clean_error()
                self.server_xarm_api.clean_warn()
                self.server_xarm_api.motion_enable(True)
                self.server_xarm_api.set_mode(0)
                self.server_xarm_api.set_state(state=0)
                time.sleep(0.1)
                self.server_xarm_api.set_servo_angle(
                    angle=np.array(self.config.xarm_init_joint_values_degree),
                    speed=15, 
                    wait=True,
                    is_radian=False,
                )  
                #TODO 
                self.server_xarm_api.set_mode(2)
                self.server_xarm_api.set_state(state=0)
            else:
                print("❌ Fail to connect to server xArm!")

        except Exception as e:
            print(f"❌ Error in init_server_xarm: {e}")

    def init_leap_hand(self):
        try:
            # safe_leap_init_joint_values_radian only in cup.yaml
            if 'safe_leap_init_joint_values_radian' in self.config and self.config['safe_leap_init_joint_values_radian']:
                self.leap_hand = self.LeapNode(kP = float(self.config.leap_kP), 
                                        kD = float(self.config.leap_kD), 
                                        init_joint_values_radian=np.array(self.config.safe_leap_init_joint_values_radian
                                        ),
                                        enable_hand=True
                                        )
                time.sleep(1.0)
                cprint("Moving LEAP Hand to safe position...", "blue")
                self.leap_hand.set_leap(self.config.leap_init_joint_values_radian)         
            else: 
                # These gains are only provided by the syringe manipulation config.
                if 'kp_side' in self.config and self.config['kp_side'] and\
                    'kd_side' in self.config and self.config['kd_side'] and\
                    'kp_T_tip' in self.config and self.config['kp_T_tip'] and\
                    'kd_T_tip' in self.config and self.config['kd_T_tip']:
                    self.leap_hand = self.LeapNode(kP = float(self.config.leap_kP), 
                                            kD = float(self.config.leap_kD), 
                                            kp_side = float(self.config.kp_side),
                                            kd_side = float(self.config.kd_side),
                                            kp_T_tip = float(self.config.kp_T_tip),
                                            kd_T_tip = float(self.config.kd_T_tip),
                                            init_joint_values_radian=np.array(self.config.leap_init_joint_values_radian),
                                            enable_hand=True
                                            )
                else:
                    self.leap_hand = self.LeapNode(kP = float(self.config.leap_kP), 
                                            kD = float(self.config.leap_kD), 
                                            init_joint_values_radian=np.array(self.config.leap_init_joint_values_radian),
                                            enable_hand=True
                                            )
            self.leap_connected = self.leap_hand.connected
            if self.leap_connected:
                print("✅ Connect ot LEAP Hand!")
            else:
                print(f"❌ Fail to connect to LEAP Hand!")
        except Exception as e:
            print(f"❌ Fail to connect to LEAP Hand! Error: {e}")
            self.leap_hand = None
            self.leap_connected = False

    def arm_hand_to_home_pose(self):
        self.leap_hand.set_leap(self.config.leap_init_joint_values_radian)
        if self.config.use_mixed_tele:
            self.server_xarm_api.set_mode(0) 
            self.server_xarm_api.set_state(state=0)
            self.server_xarm_api.motion_enable(True)
            self.server_xarm_api.set_servo_angle(
                angle=np.array(self.config.xarm_init_joint_values_degree),
                speed=15,  
                wait=True,
                radius=False, 
            )
            time.sleep(0.1)
            self.server_xarm_api.set_mode(2) 
            self.server_xarm_api.set_state(state=0)
        self.xarm_api.set_mode(0) 
        self.xarm_api.set_state(state=0)
        self.xarm_api.set_servo_angle(
            angle=np.array(self.config.xarm_init_joint_values_degree),
            speed=15,  
            wait=True,
            radius=False, 
        )
        self.moving_to_max,self.moving_to_min = False,False
        if self.config.use_kin_arm_tele:
            request = {"command": "init"}
            self.socket.send_string(json.dumps(request))
        if "xarm_target_joint_values_degree_list" in self.config:
            self.target_xarm_onehot = np.zeros(len(self.config.xarm_target_joint_values_degree_list),)

        if self.config.use_keyboard_arm_tele:
            self.xarm_api.set_mode(5) 
            self.xarm_api.set_state(state=0)
        elif self.config.use_kin_arm_tele or self.config.use_mixed_tele:
            self.xarm_api.set_mode(6)
            self.xarm_api.set_state(state=0)
        if self.mode == "policy":
            self.obs_to_takein.clear()
            self.actions_to_rollout.clear()

    def init_viser_server(self):
        global server, viser_urdf

        server = viser.ViserServer()
        server.scene.add_grid("/grid", width=2, height=2, position=(0, 0, 0))
        self.server = server
        print("🚀 Viser Started")

        xarm_path = Path(self.config.robot_urdf_path)

        assert xarm_path.exists(), f"❌ Cannot find robot URDF: {xarm_path}"

        self.viser_urdf = ViserUrdf(
            server,
            urdf_or_path=xarm_path,
            load_meshes=True,
            load_collision_meshes=False,
            scale=1,
            root_node_name="/vis_robot",
        )

        self.fk_robot = ViserUrdf(
            server,
            urdf_or_path=xarm_path,
            load_meshes=True,
            load_collision_meshes=False,
            scale=1,
            root_node_name="/fk_robot"
        )
        self.fk_robot_link_names = [l.name for l in self.fk_robot._urdf.robot.links]

        print("✅ Robot URDF loaded to viser!")

        self.recording_enabled = False
        self.to_save_buffer = False

        with server.gui.add_folder("🎥 Teleop & Record",):

            self.recording_checkbox = server.gui.add_checkbox(
                "Recording Enabled",
                initial_value=self.recording_enabled
            )

            @self.recording_checkbox.on_update
            def toggle_recording(_):
                self.recording_enabled = self.recording_checkbox.value
                if self.recording_enabled:
                    print("🔴 Start Recording")
                else:
                    print("⏹️ Stop Recording")
            
            self.save_buffer_button = server.gui.add_button(
                "Save Buffer", 
            )
            
            @self.save_buffer_button.on_click
            def toggle_to_save(_):
                self.to_save_buffer = True 
            
            self.clear_buffer_button = server.gui.add_button(
                "Clear Buffer", 
            )
            @self.clear_buffer_button.on_click
            def toggle_to_clear(_):
                self.to_save_buffer = False
                self.recording_enabled = False
                self.recording_checkbox.value = False
                self.recorder_storage.clear_buffer()
       
        with server.gui.add_folder("🤖 Robot Control"):

            self.robot_control_enabled = server.gui.add_checkbox(
                "Robot Control Enabled",
                initial_value=False
            )
            @self.robot_control_enabled.on_update
            def toggle_robot_control(_):
                enabled = self.robot_control_enabled.value
                if enabled:
                    print("🟢 Robot Control Enabled")
                    self.xarm_api.set_state(state=0)
                else:
                    print("🔴 Robot Control Disabled")
                    self.xarm_api.set_state(state=3)
                    # self.saved_aux_test = 0

            self.robot_init = server.gui.add_button(
                "Init Robot")
            self.robot_init.on_click(lambda _: self.arm_hand_to_home_pose())
            
            self.server_xarm_manual = server.gui.add_button(
                "Server xArm Manual")
            @self.server_xarm_manual.on_click
            def toggle_server_xarm_manual(_):
                cprint("Setting server xArm to manual mode...", "green")
                self.server_xarm_api.clean_error()
                self.server_xarm_api.clean_warn()
                self.server_xarm_api.motion_enable(True)
                code = self.server_xarm_api.set_mode(2)
                self.server_xarm_api.set_state(state=0)

            self.read_xarm_leap_joint_button = server.gui.add_button(
                "Read Joints", 
            )
            @self.read_xarm_leap_joint_button.on_click
            def read_xarm_leap_joint(_):
                if self.xarm_connected:
                    _,joints = self.xarm_api.get_servo_angle(is_radian=False)
                    cprint(f"🤖 xArm Joint States (radian):\n {joints}","cyan")
                if self.leap_connected:
                    leap_joint_states = self.leap_hand.read_pos()
                    cprint(f"🤖 LEAP Hand Joint States (radian):\n {leap_joint_states}","magenta")
       
        # cam visual
        self.img_gui_handles = {}
        self.depth_gui_handles = {}
        with server.gui.add_folder("📷 Cameras" ,expand_by_default=False):
            for cam in self.camera_names:
                init_img = np.zeros((240, 320, 3), dtype=np.uint8)
                self.img_gui_handles[cam] = server.gui.add_image(
                    init_img,
                    label=f"{cam} image"
                )
        with server.gui.add_folder("🎥 Realsense RGB", expand_by_default=False):
            for dev in self.rs_camera_names:
                init_img = np.zeros((240, 320, 3), dtype=np.uint8)
                self.img_gui_handles[f"rs_rgb_{dev}"] = server.gui.add_image(
                    init_img,
                    label=f"{dev} RGB"
                )
        print("✅ GUI controls added to viser!")

    def init_zmq(self):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(self.config.zmq_addr)

    def get_leap_joint_values_from_zmq(self):
        request = {"command": "get_leap_joint_values"}
        self.socket.send_string(json.dumps(request))
        reply = self.socket.recv_string()
        self.target_rw_leap_joints = json.loads(reply)["angles"]

        # get current arm joint values from the follower arm 
        self.target_viser_joint_values = deepcopy(self.viser_joint_values)
        self.target_viser_joint_values[7:] = np.asarray(deepcopy(self.target_rw_leap_joints))
        return self.target_rw_leap_joints
    
    def get_all_joint_values_from_zmq(self):
        # all are in radian 
        request = {"command": "get_all_joint_values"}
        self.socket.send_string(json.dumps(request))
        reply = self.socket.recv_string()
        all_joint_values = json.loads(reply)["angles"]

        self.target_rw_xarm_joints_radian = all_joint_values[:7]
        self.target_rw_leap_joints = all_joint_values[7:23]
        self.target_rw_joint_values = all_joint_values
        self.target_viser_joint_values = np.concatenate([self.target_rw_xarm_joints_radian, np.asarray(deepcopy(self.target_rw_leap_joints))])

        return self.target_rw_xarm_joints_radian, self.target_rw_leap_joints
    
    def get_action_from_keyboard(self):
        # TODO: only set the xyz translation speed now 
        action_list = self.keyboard_tele.get_current_actions()
        trans_speed = np.zeros(3,)
        target_joint_degrees = None

        if "x-" in action_list:
            trans_speed[0] -= self.config.keyboard_tele_speed_scale[0]
        if "y-" in action_list:
            trans_speed[1] -= self.config.keyboard_tele_speed_scale[1]
        if "z-" in action_list:
            trans_speed[2] -= self.config.keyboard_tele_speed_scale[1]
        if "x+" in action_list:
            trans_speed[0] += self.config.keyboard_tele_speed_scale[0]
        if "y+" in action_list:
            trans_speed[1] += self.config.keyboard_tele_speed_scale[1]
        if "z+" in action_list:
            trans_speed[2] += self.config.keyboard_tele_speed_scale[2]

        if "save buffer" in action_list:
            self.to_save_buffer = True
            print("🟢 [Keyboard] Set to save buffer")

        if "Init Robot" in action_list:
            self.arm_hand_to_home_pose()
            print("🟢 [Keyboard] Reset to home pose")

        if "clear buffer" in action_list:
            self.to_save_buffer = False
            self.recording_enabled = False
            self.recording_checkbox.value = False
            self.recorder_storage.clear_buffer()
            print("🟢 [Keyboard] Clear buffer")

        if "Recording Enabled" in action_list:
            self.recording_checkbox.value = True
            print("🔴 [Keyboard] Start Recording")
        elif "Recording Disabled" in action_list:
            self.recording_checkbox.value = False
            print("⏹️ [Keyboard] Stop Recording")

        if "Robot Control Enabled" in action_list:
            self.robot_control_enabled.value = True
            print("🟢 [Keyboard] Robot Control Enabled")
            self.xarm_api.set_state(state=0)
        elif "Robot Control Disabled" in action_list:
            self.robot_control_enabled.value = False
            print("🔴 [Keyboard] Robot Control Disabled")
            self.xarm_api.set_state(state=3)

        if "xarm_target_joint_values_degree_list" in self.config:
            n_targets = len(self.config.xarm_target_joint_values_degree_list)
            
            for i in range(n_targets):
                key = str(i)
                if key in action_list:
                    trans_speed = None
                    self.target_xarm_onehot = np.zeros(n_targets,)
                    target_joint_degrees = self.config.xarm_target_joint_values_degree_list[i]
                    self.target_xarm_onehot[i] = 1.0
                    break

        return trans_speed, target_joint_degrees
    
    def get_target_pose_from_keyboard(self):
        target_pose_id = self.keyboard_tele.get_current_actions()
        if len(target_pose_id) > 0 and int(target_pose_id[0]) in self.config["target_pose"] :
                target_pose = self.config["target_pose"][int(target_pose_id[0])]
                return target_pose
        else:
            return None

    def send_teleop_actions_to_robots(self, xarm_joints=None, leap_joints=None, xarm_delta_pose=None):
        # all in radian
        if leap_joints is not None and self.leap_hand and self.leap_hand.connected and self.robot_control_enabled.value:
            self.leap_hand.set_leap(leap_joints)
    
        if xarm_joints is not None and self.xarm_connected and self.robot_control_enabled.value:
            xarm_degrees = np.rad2deg(xarm_joints)
            self.xarm_api.set_mode(6)
            self.xarm_api.set_state(state=0)
            self.xarm_api.set_servo_angle(
                angle=xarm_degrees.tolist(),
                speed=15, 
                wait=False,
                is_radian=False
            )

        if xarm_delta_pose is not None and self.xarm_connected and self.robot_control_enabled.value:
            xarm_degrees = np.rad2deg(xarm_delta_pose)
            self.xarm_api.set_servo_angle(
                angle=xarm_degrees.tolist(),
                speed=15, 
                wait=False,
                is_radian=False
            )

    def get_ot(self):# observation t time
        
        ot_dict = {}
        # get camera ot 
        self.current_rs_frames = self.realsense_manager.capture_frames(img_size=(self.config.W_resize, self.config.H_resize))

        self.current_camera_frames, origin_frames = self.camera_manager.get_frames(img_size=(self.config.W_resize, self.config.H_resize)) 
        current_rs_images_array = np.array([self.current_rs_frames[rs_name]["color"] for rs_name in self.rs_camera_names]).transpose(0, 3, 1, 2)  # (n_cam, 3, H, W)
        current_rs_depth_array = np.array([self.current_rs_frames[rs_name]["depth"] for rs_name in self.rs_camera_names])  # (n_cam, H, W)
       
        current_images_array = np.array([self.current_camera_frames[cam_name] for cam_name in self.camera_names]).transpose(0, 3, 1, 2)  # (n_cam, 3, H, W)

        current_center_tag_T, current_center_tag_can_use = self.tag_detector.get_tag_pose(origin_frames) 

        for cam in self.camera_names: 
            img = self.current_camera_frames[cam]
            self.img_gui_handles[cam].image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        for dev in self.rs_camera_names: 
            img = self.current_rs_frames[dev]["color"]
            self.img_gui_handles[f"rs_rgb_{dev}"].image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ot_dict["current_center_tag_T"] = deepcopy(current_center_tag_T)
        ot_dict["current_center_tag_can_use"] = deepcopy(current_center_tag_can_use)

        ot_dict["current_rgb_images"] = deepcopy(current_images_array[:, ::-1])  # to RGB
        ot_dict["current_rs_images"] = deepcopy(current_rs_images_array[:, ::-1])  # to RGB
        ot_dict["current_rs_depth"] = deepcopy(current_rs_depth_array)
       
        ot_slice_indices = self.config.keyboard_keep_obs_idx if self.config.use_keyboard_arm_tele else self.config.kin_keep_obs_idx 
        ot_dict["current_joint_values"] = deepcopy(slice_with_list(self.rw_joint_values, ot_slice_indices))
        return ot_dict

    def teleop_prestep(self):
        # get ot 
        target_rw_xarm_joints_radian, target_trans_speed,target_rw_leap_joints, target_viser_joints,ot_dict = None, None, None,None, None

        if self.mode == "record" and self.recording_enabled:
                ot_dict = self.get_ot()
        
        # get and not execute at
        if self.config.use_keyboard_arm_tele:
            # ee translation control 
            target_trans_speed, target_joint_degrees = self.get_action_from_keyboard()
            
            if target_joint_degrees is not None:
                target_rw_xarm_joints_radian = np.deg2rad(np.array(target_joint_degrees))
            # leap control 
            target_rw_leap_joints = self.get_leap_joint_values_from_zmq()
        elif self.config.use_kin_arm_tele:
            target_rw_xarm_joints_radian, target_rw_leap_joints = self.get_all_joint_values_from_zmq()
        elif self.config.use_mixed_tele:
            target_rw_xarm_joints_radian, target_rw_leap_joints = self.get_all_joint_values_from_zmq()
                
        return target_rw_xarm_joints_radian, target_trans_speed, target_rw_leap_joints, ot_dict

    def teleop_step(self, target_rw_xarm_joints_radian, target_trans_speed, target_rw_leap_joints, ot_dict):

        if self.config.use_keyboard_arm_tele:

            if "xarm_target_joint_values_degree_list" in self.config and target_rw_xarm_joints_radian is not None:
                cprint("keyboard teleop with xarm joint position control", "yellow")
                cprint(target_rw_xarm_joints_radian, "yellow")
                self.send_teleop_actions_to_robots(xarm_joints=target_rw_xarm_joints_radian, leap_joints=target_rw_leap_joints)

            else:
                # normal xyz teleop 
                target_speed = np.concatenate([target_trans_speed, np.zeros(3,)])

                if np.count_nonzero(target_speed) > 0:
                    code = self.xarm_api.vc_set_cartesian_velocity(target_speed, is_radian=None, is_tool_coord=False, duration=self.record_interval * 1.5) 

                self.send_teleop_actions_to_robots(leap_joints=target_rw_leap_joints)

        elif self.config.use_kin_arm_tele or self.config.use_mixed_tele:
            self.send_teleop_actions_to_robots(xarm_joints=target_rw_xarm_joints_radian, leap_joints=target_rw_leap_joints)
    
    def record_step(self, target_rw_xarm_joints_radian, target_trans_speed, target_rw_leap_joints, ot_dict):
        if self.recording_enabled and ot_dict is not None:
            at_dict_to_record = {}

            if self.config.use_keyboard_arm_tele:
                if "xarm_target_joint_values_degree_list" in self.config:
                    full_action = np.concatenate([self.target_xarm_onehot, target_rw_leap_joints])
                else: 
                    full_action = np.concatenate([target_trans_speed, target_rw_leap_joints])

            elif self.config.use_kin_arm_tele or self.config.use_mixed_tele:
                full_action = np.concatenate([target_rw_xarm_joints_radian, target_rw_leap_joints])
            sliced_action = slice_with_list(full_action, self.config.keyboard_keep_action_idx if self.config.use_keyboard_arm_tele else self.config.kin_keep_action_idx)
            at_dict_to_record["target_action"] = deepcopy(sliced_action)

            self.recorder_storage.append_buffer(at_dict_to_record)
            self.recorder_storage.append_buffer(ot_dict)

        if self.recording_enabled and self.to_save_buffer:
                self.recorder_storage.save_recordings(
                    other_payload={
                        "camera_names": self.camera_names,
                        "realsense_names": self.rs_camera_names,
                        "link_names": self.fk_robot_link_names,
                    }
                )
                self.to_save_buffer = False
                self.recording_enabled = False
                self.recording_checkbox.value = False

    def policy_prestep(self):
        if not self.robot_control_enabled.value:
            self.first_time = True
            self.target_xarm_onehot = None
            if "xarm_target_joint_values_degree_list" in self.config:
                self.target_xarm_onehot = np.zeros(len(self.config.xarm_target_joint_values_degree_list),)

        if self.mode == "policy" and self.recording_enabled and self.robot_control_enabled.value:
            self.current_rs_frames = self.realsense_manager.capture_frames(img_size=(self.config.W_resize, self.config.H_resize))

        if self.robot_control_enabled.value :

            ot_dict = self.get_ot()
            ot_tensor_dict = {}
            ot_tensor_dict["obs/rgb_images"] = (ot_dict["current_rgb_images"] / 255.0)[np.newaxis, np.newaxis, ...]  # (To=1, nv, 3, h, w)
            self.tag_pose_history.get_new_tag_pose(ot_dict["current_center_tag_T"],first_frame=self.first_time)
            split_tag_pose = self.tag_pose_history.get_padded_history() # len(ot_dict['current_center_tag_T'])
            ot_tensor_dict["obs/rs_rgb_images"] = (ot_dict["current_rs_images"] / 255.0)[np.newaxis, np.newaxis, ...]  # (To=1, nv, 3, h, w)
            if "current_transforms" in ot_dict:
                ot_tensor_dict["obs/current_transforms"] = ot_dict["current_transforms"][np.newaxis, np.newaxis, ...]  # (To=1, n_link, 4, 4)
            else:
                n_links = len(getattr(self, "fk_robot_link_names", []))
                if n_links == 0:
                    n_links = 1
                ot_tensor_dict["obs/current_transforms"] = np.tile(np.eye(4, dtype=np.float32), (n_links, 1, 1))[np.newaxis, np.newaxis, ...]
            ot_tensor_dict["obs/all_qpos"] = ot_dict["current_joint_values"][np.newaxis, np.newaxis, ...]  # (To=1, n_joint)
            ot_tensor_dict["obs/tag_ori"] = split_tag_pose[np.newaxis, np.newaxis, ...]

            ot_tensor_dict = {k: torch.tensor(v, dtype=torch.float32).to(self.policy.device) for k, v in ot_tensor_dict.items() if isinstance(v, np.ndarray)}
            self.obs_to_takein.append(ot_tensor_dict)

            if len(self.actions_to_rollout) == 0:
                ot_tensor_dict_take_in = concat_and_pad_obs(
                    self.obs_to_takein,
                    To=self.n_obs_steps,
                )

                predictions = self.policy.predict_action(
                    ot_tensor_dict_take_in,
                    action_key=self.action_key,
                )
                pred_actions_to_rollout = predictions["actions"].detach().cpu().numpy()[0]  # (Ta, ndof)
                for a in pred_actions_to_rollout:
                    self.actions_to_rollout.append(a)
            self.first_time = False
            
            return ot_dict
    
    def policy_step(self):
        target_rw_xarm_joints_radian = None
        target_trans_speed = None 
        target_rw_leap_joints = None 
        if self.robot_control_enabled.value and len(self.actions_to_rollout) > 0:
            action = self.actions_to_rollout.popleft()
            if self.config.use_keyboard_arm_tele:
                if "xarm_target_joint_values_degree_list" in self.config:
                    xarm_state_list_len = len(self.config.xarm_target_joint_values_degree_list)
                    full_action = np.zeros(xarm_state_list_len+16,)
                    full_action = refill_full_list_with_slice_indices(full_action, self.config.keyboard_keep_action_idx, action)
                    target_rw_leap_joints = full_action[xarm_state_list_len:]
                    target_rw_leap_joints = disable_and_refill(
                        target_rw_leap_joints, self.config.tele_leap_disabled_idx,
                        self.config.leap_init_joint_values_radian,
                        dim_to_slice=-1, refill_values=0)                    
                    xarm_state_list = full_action[:xarm_state_list_len]
                    for i in range(xarm_state_list_len):
                        if xarm_state_list[i] > 0.5:
                            target_rw_xarm_joints_radian = np.deg2rad(np.array(self.config.xarm_target_joint_values_degree_list[i]))
                            break
                    self.send_teleop_actions_to_robots(leap_joints=target_rw_leap_joints, xarm_joints=target_rw_xarm_joints_radian)
                else:
                # full_action is always arm_translation_speed + absolute_leap_joints
                    full_action = np.zeros(3+16,)
                    if self.action_form == "absolute_joint_values":
                        full_action[3:] = 0 
                        full_action = refill_full_list_with_slice_indices(full_action, self.config.keyboard_keep_action_idx, action)
                        target_speed = np.concatenate([full_action[:3], np.zeros(3)])
                        target_rw_leap_joints = full_action[3:]
                        target_rw_leap_joints = disable_and_refill(
                            target_rw_leap_joints, self.config.tele_leap_disabled_idx,
                            self.config.leap_init_joint_values_radian,
                            dim_to_slice=-1, refill_values=0)
                    if np.count_nonzero(target_speed) > 0:
                        code = self.xarm_api.vc_set_cartesian_velocity(target_speed, is_radian=None, is_tool_coord=False, duration=self.record_interval * 1.1) 
                    self.send_teleop_actions_to_robots(leap_joints=target_rw_leap_joints)
            elif self.config.use_kin_arm_tele or self.config.use_mixed_tele:
                full_action = np.zeros(23,)
                full_action[7:] = 0 
                full_action = refill_full_list_with_slice_indices(full_action, self.config.kin_keep_action_idx, action)

                target_rw_xarm_joints_radian = full_action[:7]
                target_rw_leap_joints = full_action[7:]

                self.send_teleop_actions_to_robots(xarm_joints=target_rw_xarm_joints_radian, leap_joints=target_rw_leap_joints)
        return target_rw_xarm_joints_radian, target_trans_speed, target_rw_leap_joints
    
    def update_robot_states_and_viser_urdf(self):
        # update current configuration vis
        _, xarm_rw_joints_radian = self.xarm_api.get_servo_angle(is_radian=True)
        self.xarm_rw_joints_radian = np.array(xarm_rw_joints_radian)

        self.leap_rw_joints = np.array(self.leap_hand.read_pos()) 
        self.leap_rw_joints_viser = self.leap_rw_joints
        self.viser_joint_values = np.concatenate([self.xarm_rw_joints_radian, self.leap_rw_joints_viser])
        self.rw_joint_values = np.concatenate([self.xarm_rw_joints_radian, self.leap_rw_joints])
        self.viser_urdf.update_cfg(self.viser_joint_values)
        
    def update_target_viser_urdf(self):
        target_viser_leap_joints = np.asarray(deepcopy(self.target_rw_leap_joints))
        if self.config.use_keyboard_arm_tele:
            self.fk_robot.update_cfg(np.concatenate([self.xarm_rw_joints_radian, target_viser_leap_joints]))
        elif self.config.use_kin_arm_tele or self.config.use_mixed_tele:
            self.fk_robot.update_cfg(np.concatenate([self.target_rw_xarm_joints_radian, target_viser_leap_joints]))
            
    def prestep_sleep(self):
        now = time.monotonic()
        cur_idx = int((now - self.t_start) // self.record_interval)
        if cur_idx > self.frame_idx:
            missed = cur_idx - self.frame_idx
            late_by = now - (self.t_start + (self.frame_idx + 1) * self.record_interval)
            if late_by > 0:
                cprint(
                    f"[TIMING OVERRUN] Missed {missed} frame(s); late by {late_by:.4f}s. "
                    f"Resync to frame_idx={cur_idx}.",
                    "red", attrs=["bold"]
                )
            self.frame_idx = cur_idx  # resync so the next deadline is in the future
        t_cycle_end = self.t_start + (self.frame_idx + 1) * self.record_interval
        t_obs = t_cycle_end - self.action_interval
        precise_wait(t_obs)

    def poststep_sleep(self):
        t_cycle_end = self.t_start + (self.frame_idx + 1) * self.record_interval
        precise_wait(t_cycle_end)

    def run(self):
        self.t_start = time.monotonic()
        # Warm up cameras only when camera observations are enabled.
        if self.use_cam_obs:
            warm_step = 0
            while True:
                current_camera_frames, origin_frames = self.camera_manager.get_frames(img_size=(self.config.W_resize, self.config.H_resize)) 
                current_center_tag_T, current_center_tag_can_use = self.tag_detector.get_tag_pose(origin_frames)
                warm_step += 1
                if all(current_center_tag_can_use.values()) and warm_step >= 50:
                    break
                time.sleep(0.01)
            cprint("✅ Camera warm-up done.", "green")
        while True:
            self.update_robot_states_and_viser_urdf()

            if self.use_action_from_teleop:
                t_begin_teleop_obs = time.monotonic()
                target_rw_xarm_joints_radian, target_trans_speed, target_rw_leap_joints, ot_dict = self.teleop_prestep()
                precise_wait(t_begin_teleop_obs + self.obs_interval)

                t_begin_teleop_action = time.monotonic()
                self.teleop_step(target_rw_xarm_joints_radian, 
                                 target_trans_speed, 
                                 target_rw_leap_joints,
                                 ot_dict)
                precise_wait(t_begin_teleop_action + self.action_interval)

                self.update_target_viser_urdf()
                
                self.record_step(target_rw_xarm_joints_radian, target_trans_speed, target_rw_leap_joints, ot_dict)
            else: 
                self.t_begin_policy_obs = time.monotonic()
                ot_dict = self.policy_prestep()
                precise_wait(self.t_begin_policy_obs + self.obs_interval)
                self.t_begin_policy_action = time.monotonic()
                target_rw_xarm_joints_radian, target_trans_speed, target_rw_leap_joints = self.policy_step()
                precise_wait(self.t_begin_policy_action + self.action_interval)

@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.joinpath("configs")),
    config_name="coin_standing.yaml",
)
def main(config):
    cprint(OmegaConf.to_yaml(config), "grey")

    if config.name == "syringe_manipulation":
        leap_module = importlib.import_module("leap_hand_utils.leapnode_syringe")
    else:
        leap_module = importlib.import_module("leap_hand_utils.leapnode")
    LeapNode = getattr(leap_module, "LeapNode")

    tele = ConTacTeleop(config, LeapNode=LeapNode)
    tele.run()

if __name__ == "__main__":
    main()
