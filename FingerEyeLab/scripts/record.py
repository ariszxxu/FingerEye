import time
import zmq
import json
import sys
import numpy as np
import viser
import math
import wandb 
import argparse
from pathlib import Path
from tqdm import tqdm
from termcolor import cprint
from collections import deque
from viser.extras import ViserUrdf
from utils.recoder import RecorderStorage
from utils.utils import slice_with_list
from isaaclab.app import AppLauncher
from pathlib import Path
from utils.utils import configure_seed, draw_status_overlay
FILE_PATH = Path(__file__).resolve()
DIR_PATH = FILE_PATH.parent.parent  

# add argparse arguments
parser = argparse.ArgumentParser(description="Zero agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=0, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Random seed for the environment.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# randomlize option
parser.add_argument("--no_rand_all", action="store_true")
parser.add_argument("--no_rand_light", action="store_true")
parser.add_argument("--no_rand_coin", action="store_true")
parser.add_argument("--no_rand_camera", action="store_true")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

configure_seed(args_cli.seed, torch_deterministic=True)

# launch omniverse app
app_launcher = AppLauncher(args_cli)
configure_seed(args_cli.seed, torch_deterministic=True)

simulation_app = app_launcher.app

import hydra
from omegaconf import DictConfig, OmegaConf
import gymnasium as gym
import torch
import numpy as np


import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import fingereye.tasks  # noqa: F401
sys.argv = [sys.argv[0]] + hydra_args

class SimulatorRecorder:
    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.get_init_pos()
        self.real_joint_names = self.config.real_joint_names
        self.leap_joint_names = self.config.leap_joint_names
        self.control_joint_names = self.config.control_joint_names
        self.target_indices =  [self.real_joint_names.index(joint_name) for joint_name in self.control_joint_names]
        self.control_indices = [self.leap_joint_names.index(joint_name) for joint_name in self.control_joint_names]

    def get_init_pos(self):
        self.init_pos = np.zeros(23)

        xarm_init_joint_values_degree = np.deg2rad(np.asarray(self.config.xarm_init_joint_values_degree))
        leap_init_joint_values_radian = np.asarray(self.config.leap_init_joint_values_radian) - np.pi
        
        self.init_pos[:7] = xarm_init_joint_values_degree
        self.init_pos[7:23] = leap_init_joint_values_radian

    def init_zmq(self):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(self.config.zmq_addr)
    
    def get_all_joint_values_from_zmq(self):
        # all are in radian 
        request = {"command": "get_all_joint_values"}
        self.socket.send_string(json.dumps(request))
        reply = self.socket.recv_string()
        target_viser_joint_values = json.loads(reply)["angles"]

        return target_viser_joint_values

    def get_leap_joint_values_from_zmq(self):
        """Get LEAP hand joint values from ZMQ"""
        request = {"command": "get_leap_joint_values"}
        self.socket.send_string(json.dumps(request))
        reply = self.socket.recv_string()
        target_rw_leap_joints = json.loads(reply)["angles"]
        target_viser_joint_values = np.asarray(target_rw_leap_joints)
        return target_viser_joint_values

    def init_viser_server(self):
        """Initialize Viser visualization server"""
        global server

        server = viser.ViserServer()
        server.scene.add_grid("/grid", width=2, height=2, position=(0, 0, 0))
        self.server = server
        print("🚀 Viser Started")

        xarm_path = Path(DIR_PATH / "assets/xarm7_leap_right/xarm7/xarm7_leap_right_rwv2.urdf")
        assert xarm_path.exists(), f"❌ Cannot find robot URDF: {xarm_path}"
        # Viser urdf for target robot
        self.target_urdf = ViserUrdf(
            server,
            urdf_or_path=xarm_path,
            load_meshes=True,
            load_collision_meshes=False,
            scale=1.0,
            root_node_name="/target_robot",
        )
        self.current_urdf = ViserUrdf(
            server,
            urdf_or_path=xarm_path,
            load_meshes=True,
            load_collision_meshes=False,
            scale=1.0,
            root_node_name="/current_robot",
        )
        self.fk_robot_link_names = [l.name for l in self.target_urdf._urdf.robot.links]
        print(f"✅ Robot URDF loaded to viser (original size)!")
        # Update viser with config values
        self.target_urdf.update_cfg(self.init_pos) 
        self.current_urdf.update_cfg(self.init_pos)
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

        self.robot_control_enabled = False
        
        with server.gui.add_folder("🤖 Simulation Control"):
            self.control_checkbox = server.gui.add_checkbox(
                "Enable Control",
                initial_value=self.robot_control_enabled
            )
            
            @self.control_checkbox.on_update
            def toggle_control(_):
                self.robot_control_enabled = self.control_checkbox.value
                if self.robot_control_enabled:
                    cprint("🟢 Robot Control Enabled", "green")
                else:
                    cprint("🔴 Robot Control Disabled", "red")

            self.reset_button = server.gui.add_button("Reset Simulation")
            @self.reset_button.on_click
            def reset_sim(_):
                self.reset_robot()
                cprint("🔄 Simulation Reset", "yellow")

        self.fingertip_cam_names = list(self.config.camera_names) # TODO
        self.img_gui_handles = {}
        init_img = np.zeros((480, 640, 3), dtype=np.uint8)
        self.img_gui_handles["third_view"] = server.gui.add_image(
                    init_img,
                    label=f"third_view image"
                )
        print("✅ GUI controls added to viser!")

    def update_robot_states_and_current_urdf(self):
        cur_positions = self.env.actuated_dof_pos[0].cpu().numpy()
        self.current_urdf.update_cfg(cur_positions)

    def update_target_viser_urdf(self, target_viser_joint):
        joint_values = self.init_pos
        joint_values[self.target_indices] = target_viser_joint
        self.target_urdf.update_cfg(joint_values)
        
    def reset_robot(self):
        self.env.reset()
        self.current_urdf.update_cfg(self.init_pos)
        self.target_urdf.update_cfg(self.init_pos)

    def get_action_from_zmq(self):
        """get target joint values and ot"""
        if hasattr(self.config, 'use_kin_arm_tele') and self.config.use_kin_arm_tele:
            target_viser_joints_np = self.get_all_joint_values_from_zmq()[self.control_indices]
        if hasattr(self.config, 'use_keyboard_arm_tele') and self.config.use_keyboard_arm_tele:
            target_viser_joints_np = self.get_leap_joint_values_from_zmq()[self.control_indices]
        target_viser_joints_tensor = torch.from_numpy(target_viser_joints_np).to(self.env.device, dtype=torch.float32).unsqueeze(0).repeat(self.env.num_envs, 1)
        return target_viser_joints_tensor, target_viser_joints_np

    def record_step(self, target_viser_joints, obs):
        if self.recording_enabled and target_viser_joints is not None:

            del obs["third_view"]

            obs_np = {
                k: v.detach().cpu().numpy() if torch.is_tensor(v) else v
                for k, v in obs.items()
            }
            self.recorder_storage.append_buffer(obs_np)
            at_dict_to_record = {}
            full_action = target_viser_joints
            at_dict_to_record["target_action"] = full_action
            self.recorder_storage.append_buffer(at_dict_to_record)
        if self.recording_enabled and self.to_save_buffer:
            self.recorder_storage.save_recordings(
                other_payload={
                    "link_names": self.fk_robot_link_names,
                }
            )
            self.to_save_buffer = False
            self.recording_enabled = False
            self.recording_checkbox.value = False

    def run_record(self):
        self.recorder_storage = RecorderStorage()
        self.init_zmq()
        self.init_viser_server()
        target_viser_joints_tensor = None  # (num_envs, action_dim) tensor

        while simulation_app.is_running():
            self.update_robot_states_and_current_urdf()
            if self.robot_control_enabled:  # Run teleop and record each control cycle.
                target_viser_joints_tensor, target_viser_joints_np = self.get_action_from_zmq()
                if target_viser_joints_tensor is not None:
                    obs, _, _, _, _ = self.env.step(target_viser_joints_tensor) 
                    self.img_gui_handles["third_view"].image = obs["third_view"][0].cpu().numpy()
                    self.update_target_viser_urdf(target_viser_joints_np)
                    self.record_step(target_viser_joints_np, obs)

    def load_policy(self):
        from fingereye.workspaces.workspace import TrainWorkspace
        assert Path(self.config.eval_ckpt_path).exists(), f"❌ Cannot find eval checkpoint: {self.config.eval_ckpt_path}"
        if not hasattr(self.config, 'eval_ckpt_config_path') or self.config.eval_ckpt_config_path is None:
            # try to find the config file in the same dir as the checkpoint
            self.config.eval_ckpt_config_path = Path(self.config.eval_ckpt_path).parent.parent / ".hydra" / "config.yaml"
        workspace_config = OmegaConf.load(self.config.eval_ckpt_config_path)
        self.workspace_config = workspace_config
        workspace_config.eval_ckpt_path = self.config.eval_ckpt_path
        cprint(OmegaConf.to_yaml(workspace_config), "grey")
        workspace = TrainWorkspace(workspace_config)
        self.policy = workspace.run_eval()
        self.actions_to_rollout = deque(maxlen=workspace_config.n_action_steps)
        self.obs_to_takein = deque(maxlen=workspace_config.n_obs_steps)
        self.replay_buffer_keys = workspace_config.setting.dataset.replay_buffer_keys
        self.n_obs_steps = workspace_config.n_obs_steps
        # get the action form 
        self.stage = workspace_config.training.get("stage", "joint")
        self.action_key = "actions/original_actions"
        self.action_form = "absolute_joint_values"
        self.eval_policy_batch_size = int(getattr(self.config, "eval_policy_batch_size", 0) or 0)
        if self.eval_policy_batch_size > 0:
            print(f"[INFO] Eval policy batch size: {self.eval_policy_batch_size}")

    def image2radio(self, images):
        b, nobs, nv, c, h, w = images.shape
        images_stack = images.reshape(-1, c, h, w)
        with torch.no_grad():
            feat_grid, feat_summary = self.policy.image_encoder.vit.get_feature_grid(images_stack, return_processed_img=False)  # (B*nv, Hp, Wp, D_vit)
            feat_summary = feat_summary.view(b, nobs, nv, -1)  # (B, nobs, nv, D_vit)
        return feat_summary


    def _predict_action_once(self, obs_dict):
        if self.stage != "joint":
            return self.policy.predict_action(
                obs_dict,
                action_key=self.action_key,
                stage=self.stage,
            )
        return self.policy.predict_action(
            obs_dict,
            action_key=self.action_key,
        )

    def _predict_action_batched(self, obs_dict):
        batch_size = int(getattr(self, "eval_policy_batch_size", 0) or 0)
        first_tensor = next((value for value in obs_dict.values() if torch.is_tensor(value)), None)
        if first_tensor is None:
            return self._predict_action_once(obs_dict)

        n_envs = int(first_tensor.shape[0])
        if batch_size <= 0 or batch_size >= n_envs:
            return self._predict_action_once(obs_dict)

        action_chunks = []
        action_pred_chunks = []
        attention_chunks = []
        for start in range(0, n_envs, batch_size):
            end = min(start + batch_size, n_envs)
            chunk_obs = {
                key: value[start:end] if torch.is_tensor(value) and value.shape[0] == n_envs else value
                for key, value in obs_dict.items()
            }
            chunk_pred = self._predict_action_once(chunk_obs)
            action_chunks.append(chunk_pred["actions"])
            if "actions_pred" in chunk_pred:
                action_pred_chunks.append(chunk_pred["actions_pred"])
            if torch.is_tensor(chunk_pred.get("attention_weights")):
                attention_chunks.append(chunk_pred["attention_weights"])
            del chunk_pred
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result = {"actions": torch.cat(action_chunks, dim=0)}
        if len(action_pred_chunks) == len(action_chunks):
            result["actions_pred"] = torch.cat(action_pred_chunks, dim=0)
        if len(attention_chunks) == len(action_chunks):
            result["attention_weights"] = torch.cat(attention_chunks, dim=0)
        return result

    def get_action_from_policy(self, ot_dict):
        ######################
        ot_tensor_dict = {}
        if "current_rgb_images" in ot_dict and ot_dict["current_rgb_images"] is not None:
            current_rgb_images = (ot_dict["current_rgb_images"] / 255.0).unsqueeze(1).permute(0, 1, 2, 5, 3, 4) #(ne, To=1, nv, h, w, 3) -> (ne, To=1, nv, 3, h, w)
            # ot_tensor_dict["obs/rgb_images_radio"] = self.image2radio(current_rgb_images)
            ot_tensor_dict["obs/rgb_images"] = current_rgb_images
        if "current_rs_images" in ot_dict and ot_dict["current_rs_images"] is not None:
            current_rs_images = (ot_dict["current_rs_images"] / 255.0).unsqueeze(1).permute(0, 1, 2, 5, 3, 4) # (ne, To=1, nv, 3, h, w)
            # ot_tensor_dict["obs/rs_rgb_images_radio"] = self.image2radio(current_rs_images)
            ot_tensor_dict["obs/rs_rgb_images"] = current_rs_images
        
        if "current_joint_values" in ot_dict and ot_dict["current_joint_values"] is not None:
            current_joint_values = ot_dict["current_joint_values"]
            if "obs/all_qpos" in self.replay_buffer_keys:
                all_qpos = current_joint_values
                expected_ds = int(OmegaConf.select(self.workspace_config, "setting.ds", default=all_qpos.shape[-1]))
                if expected_ds == all_qpos.shape[-1] + 3:
                    if "hp_eef_pose" not in ot_dict or ot_dict["hp_eef_pose"] is None:
                        raise RuntimeError("Policies using 11-D obs/all_qpos require obs['hp_eef_pose'] for EEF xyz.")
                    all_qpos = torch.cat((ot_dict["hp_eef_pose"][:, :3], all_qpos), dim=-1)
                ot_tensor_dict["obs/all_qpos"] = all_qpos.unsqueeze(1) # (ne, To=1, n_joint)
        if "current_rgb_camera_poses" in ot_dict and ot_dict["current_rgb_camera_poses"] is not None and "obs/current_rgb_camera_poses" in self.replay_buffer_keys:
            ot_tensor_dict["obs/current_rgb_camera_poses"] = ot_dict["current_rgb_camera_poses"].unsqueeze(1) # (ne, To=1, nv, 7)
        if "current_rs_camera_poses" in ot_dict and ot_dict["current_rs_camera_poses"] is not None and "obs/current_rs_camera_poses" in self.replay_buffer_keys:
            ot_tensor_dict["obs/current_rs_camera_poses"] = ot_dict["current_rs_camera_poses"].unsqueeze(1) # (ne, To=1, nv, 7)

        self.obs_to_takein.append(ot_tensor_dict)

        if len(self.actions_to_rollout) == 0:
            ot_tensor_dict_take_in = self.concat_and_pad_obs(
                self.obs_to_takein,
                To=self.n_obs_steps,
            )
            predictions = self._predict_action_batched(ot_tensor_dict_take_in)
            pred_actions_to_rollout = predictions["actions"]  # (b, Ta, ndof)
            pred_actions_to_rollout = pred_actions_to_rollout.permute(1, 0, 2)  # (Ta, b, ndof)
            for Ta_action in pred_actions_to_rollout:
                self.actions_to_rollout.append(Ta_action) # (b,ndof)
        return self.actions_to_rollout
    
    def run_policy(self):
        """
        Runs a single parallel rollout across all environments.
        Returns aggregated metrics and a tiled video of specific environments.
        """
        self.load_policy()
        max_steps = int(self.env.cfg.episode_length_s // self.env.cfg.sim.dt // self.env.cfg.decimation) 
        print(f"[INFO] Starting Parallel Eval (Max Steps: {max_steps})...")
        
        # -----------------------------------------------------------------------
        # 1. Configuration & Setup
        # -----------------------------------------------------------------------
        n_envs = self.env.unwrapped.num_envs
        
        # Video settings
        n_record = getattr(self.env.cfg, 'n_env_record', 10)  # Default 10
        record_interval = getattr(self.env.cfg, 'rollout_record_interval', 3) # Default 3
        
        # Select indices to record (e.g., first N or random N)
        # We clip it in case n_envs < n_record
        n_record = min(n_record, n_envs)
        record_indices = torch.arange(n_record, device=self.env.device)
        
        # Buffers
        success_flags = torch.zeros(n_envs, dtype=torch.bool, device=self.env.device)
        steps_to_success = torch.zeros(n_envs, dtype=torch.float, device=self.env.device)
        collected_frames = [] # Will store (H, W, 3) grids
        
        # Store the "last seen frame" for freezing completed envs
        # Initialize with black or first frame
        last_seen_frames = None 

        # -----------------------------------------------------------------------
        # 2. Reset & Init
        # -----------------------------------------------------------------------
        # Note: If your env is wrapped, ensure we get the raw dict back
        obs = self.env.reset(seed=self.env.cfg.seed)
        if isinstance(obs, tuple): obs = obs[0] # Handle (obs, info)

        step_count = 0
        
        # Progress bar for visual feedback
        pbar = tqdm(total=max_steps, desc="Eval Rollout")

        # -----------------------------------------------------------------------
        # 3. Rollout Loop
        # -----------------------------------------------------------------------
        configure_seed(args_cli.seed, torch_deterministic=True)
        while step_count < max_steps:
            
            # --- A. Policy Inference ---
            # Assume get_action_from_policy returns (Chunk_Size, Num_Envs, Action_Dim)
            # OR (Num_Envs, Chunk_Size, Action_Dim). 
            # We standardize to a list of (Num_Envs, Action_Dim)
            with torch.no_grad():
                self.get_action_from_policy(obs)

                
            # Check Global Stop
            if step_count >= max_steps: break

            # 1. Step Environment
            # Note: self.env.step() ALREADY handles the decimation (control_dt).
            # We do NOT need an inner loop for sim.dt here.
            action = self.actions_to_rollout.popleft()
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Handle tuple return if using Gym wrapper
            if isinstance(next_obs, tuple): next_obs = next_obs[0]
            
            # 2. Track Success
            # Success is Reward > 0.5 (assuming 0/1 reward) AND not previously successful
            current_success = (reward > 0.5).flatten().to(self.env.device)
            new_success = current_success & (~success_flags)
            
            # Mark success
            success_flags = success_flags | new_success
            # Record the step number for newly successful envs
            steps_to_success[new_success] = step_count
            
            # 3. Video Recording Logic
            if step_count % record_interval == 0:
                # Extract images: (N, H, W, 3)
                # Ensure they are on CPU/Numpy for video processing
                current_imgs = next_obs["third_view"][record_indices]
                
                if last_seen_frames is None:
                    last_seen_frames = current_imgs.clone()
                
                # Include newly successful envs once so the frozen frame shows the
                # final success-state visuals instead of the previous frame.
                active_mask = ~success_flags[record_indices] # (n_record,)
                new_success_mask = new_success[record_indices]
                update_mask = active_mask | new_success_mask
                if update_mask.any():
                    last_seen_frames[update_mask] = current_imgs[update_mask]
                
                # Convert to Numpy for processing
                frames_np = last_seen_frames.cpu().numpy() # (N_rec, H, W, 3)
                
                # We create a list to hold the processed frames for this step
                annotated_frames = []

                # Check if this is the absolute last step of the episode
                is_last_step = (step_count >= max_steps - 1)

                # Iterate over each environment being recorded
                for i, global_env_idx in enumerate(record_indices):
                    frame = frames_np[i] # Get the frozen or active frame
                    is_success = success_flags[global_env_idx].item()

                    if is_success:
                        # 1. SUCCESS: Apply Green (Always apply if success flag is True)
                        # Since the frame is "frozen" in last_seen_frames, 
                        # we just overlay the green check every time we generate the video.
                        frame = draw_status_overlay(frame, "success")
                    
                    elif is_last_step and not is_success:
                        # 2. FAILURE: Apply Red (Only on the very last frame)
                        frame = draw_status_overlay(frame, "fail")
                    
                    annotated_frames.append(frame)
                
                # Convert back to numpy array for grid creation
                frames_np = np.stack(annotated_frames)
                
                # Create Grid (e.g., 2x5 or 3x4)
                grid_frame = self._create_grid_image(frames_np)
                collected_frames.append(grid_frame)

            # Prepare for next step
            obs = next_obs
            step_count += 1
            pbar.update(1)

        pbar.close()

        # -----------------------------------------------------------------------
        # 4. Compute Metrics & Package
        # -----------------------------------------------------------------------
        
        # Success Rate: Fraction of envs that hit reward=1 at any point
        final_success_rate = success_flags.float().mean().item()
        
        # Avg Steps: Only average over those that actually succeeded
        if success_flags.any():
            avg_steps = steps_to_success[success_flags].mean().item()
        else:
            avg_steps = max_steps # Penalty if no one succeeds

        metrics = {
            "eval/success_rate": final_success_rate,
            "eval/avg_steps_to_success": avg_steps,
        }
        
        print(f"[RESULT] Success Rate: {final_success_rate*100:.1f}% | Avg Steps: {avg_steps:.1f}")
        
        # -----------------------------------------------------------------------
        # [NEW] Sanity Check: Export Video to Local Disk
        # -----------------------------------------------------------------------
        if len(collected_frames) > 0:
            import imageio
            output_video_path = Path("./eval_video.mp4")
            print(f"[INFO] Saving local debug video to {output_video_path.absolute()}...")
            
            # Stack frames: (T, H, W, 3)
            vid_np = np.stack(collected_frames, axis=0)
            
            # Ensure uint8 [0, 255] range
            if vid_np.max() <= 1.0:
                vid_np = (vid_np * 255).astype(np.uint8)
            else:
                vid_np = vid_np.astype(np.uint8)
                
            # Write to MP4 using imageio (standard for this)
            fps = 1.0 / (self.env.cfg.sim.dt * self.env.cfg.decimation)
            imageio.mimwrite(output_video_path, vid_np, fps=fps, codec="libx264")

        return metrics

    def _rollout_once(self, seed: int):
        """Run one rollout for a single seed and return rollout metrics."""
        max_steps = int(self.env.cfg.episode_length_s // self.env.cfg.sim.dt // self.env.cfg.decimation) 
        print(f"[INFO] Starting Parallel Eval (Seed={seed}")
        
        n_envs = self.env.unwrapped.num_envs

        # Video settings
        n_record = getattr(self.env.cfg, 'n_env_record', 10)
        record_interval = getattr(self.env.cfg, 'rollout_record_interval', 3)
        n_record = min(n_record, n_envs)
        record_indices = torch.arange(n_record, device=self.env.device)

        # Metrics buffers (success flag and step-to-success per env)
        success_flags = torch.zeros(n_envs, dtype=torch.bool, device=self.env.device)
        steps_to_success = torch.zeros(n_envs, dtype=torch.float, device=self.env.device)

        # Video buffers
        collected_frames = []
        last_seen_frames = None

        # Set deterministic seed
        configure_seed(seed, torch_deterministic=True)

        # Reset environment with the rollout seed
        obs = self.env.reset(seed=seed)
        if isinstance(obs, tuple):
            obs = obs[0]

        step_count = 0
        pbar = tqdm(total=max_steps, desc=f"Eval Rollout (seed={seed})")

        while step_count < max_steps:
            # --- Inference ---
            with torch.no_grad():
                self.get_action_from_policy(obs)

            if step_count >= max_steps:
                break

            action = self.actions_to_rollout.popleft()
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            if isinstance(next_obs, tuple):
                next_obs = next_obs[0]

            # --- Success tracking ---
            current_success = (reward > 0.5).flatten().to(self.env.device)
            new_success = current_success & (~success_flags)
            success_flags = success_flags | new_success
            steps_to_success[new_success] = step_count

            # --- Video recording ---
            if step_count % record_interval == 0:
                current_imgs = next_obs["third_view"][record_indices]

                if last_seen_frames is None:
                    last_seen_frames = current_imgs.clone()

                active_mask = ~success_flags[record_indices]
                new_success_mask = new_success[record_indices]
                update_mask = active_mask | new_success_mask
                if update_mask.any():
                    last_seen_frames[update_mask] = current_imgs[update_mask]

                frames_np = last_seen_frames.cpu().numpy()
                annotated_frames = []

                is_last_step = (step_count >= max_steps - 1)

                for i, global_env_idx in enumerate(record_indices):
                    frame = frames_np[i]
                    is_success = success_flags[global_env_idx].item()

                    if is_success:
                        frame = draw_status_overlay(frame, "success")
                    elif is_last_step and not is_success:
                        frame = draw_status_overlay(frame, "fail")

                    annotated_frames.append(frame)

                frames_np = np.stack(annotated_frames)
                grid_frame = self._create_grid_image(frames_np)
                collected_frames.append(grid_frame)

            obs = next_obs
            step_count += 1
            pbar.update(1)

        pbar.close()

        # Per-seed metrics
        final_success_rate = success_flags.float().mean().item()
        if success_flags.any():
            avg_steps = steps_to_success[success_flags].mean().item()
        else:
            avg_steps = max_steps

        print(f"[RESULT][seed={seed}] Success Rate: {final_success_rate*100:.1f}% | Avg Steps: {avg_steps:.1f}")

        # Save one video per seed
        if len(collected_frames) > 0:
            import imageio
            output_video_path = Path(f"./eval_video_seed{seed}.mp4")
            print(f"[INFO] Saving video for seed={seed} to {output_video_path.absolute()}...")

            vid_np = np.stack(collected_frames, axis=0)
            if vid_np.max() <= 1.0:
                vid_np = (vid_np * 255).astype(np.uint8)
            else:
                vid_np = vid_np.astype(np.uint8)

            # Use true control frequency instead of floor division to avoid fps=0.
            fps = 1.0 / (self.env.cfg.sim.dt * self.env.cfg.decimation)
            imageio.mimwrite(output_video_path, vid_np, fps=fps, codec="libx264")

        # Return this seed result for final aggregation.
        metrics = {
            "success_rate": final_success_rate,
            "avg_steps_to_success": avg_steps,
        }
        return metrics, success_flags

    def run_policy_multi_seeds(self, seeds=(0, 1, 2)):
        """
        Run multiple seeds (default: 0, 1, 2) and report:
        - overall success rate across all envs and all seeds
        - per-seed success metrics
        """
        self.load_policy()  # Load policy once and reuse for all seeds.

        n_envs = self.env.unwrapped.num_envs
        device = self.env.device

        num_seeds = len(seeds)

        # Count how many seeds each environment succeeds on.
        success_counts_per_env = torch.zeros(n_envs, dtype=torch.float32, device=device)

        per_seed_metrics = {}

        for seed in seeds:
            metrics, success_flags = self._rollout_once(seed=seed)

            # Store this seed's metrics.
            per_seed_metrics[f"seed_{seed}/success_rate"] = metrics["success_rate"]
            per_seed_metrics[f"seed_{seed}/avg_steps_to_success"] = metrics["avg_steps_to_success"]

            # Accumulate success counts for each env.
            success_counts_per_env += success_flags.float()

        # ------------ Overall success rate across envs x seeds ------------
        overall_success_rate = success_counts_per_env.sum().item() / (n_envs * num_seeds)

        # ------------ Per-env success ratio over seeds ------------

        print("=====================================================")
        print(f"[FINAL] overall_success_rate (over {num_seeds} seeds): {overall_success_rate*100:.1f}%")
        print("[FINAL] per-env success_rate over seeds:")

        metrics = {
            "overall/success_rate": overall_success_rate,
            "overall/num_seeds": num_seeds,
        }
        metrics.update(per_seed_metrics)

        return metrics

    # -----------------------------------------------------------------------
    # Helper Methods
    # -----------------------------------------------------------------------

    def _create_grid_image(self, images):
        """
        Stitches a batch of images (N, H, W, C) into a single grid image.
        """
        n, h, w, c = images.shape
        # Compute Grid size (approx square)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        
        # Canvas
        grid = np.zeros((rows * h, cols * w, c), dtype=images.dtype)
        
        for idx in range(n):
            r = idx // cols
            c_idx = idx % cols
            grid[r*h : (r+1)*h, c_idx*w : (c_idx+1)*w, :] = images[idx]
            
        return grid

    def _pack_wandb_video(self, frames):
        """
        Expects List of (H, W, 3) -> Returns WandB Video Object (T, C, H, W)
        """
        if len(frames) == 0: return None
        
        # Stack: (T, H, W, 3)
        vid_tensor = np.stack(frames, axis=0)
        
        # Transpose for WandB/Torch: (T, 3, H, W)
        vid_tensor = vid_tensor.transpose(0, 3, 1, 2)
        
        return wandb.Video(vid_tensor, fps=20, format="mp4")
    
    def concat_and_pad_obs(
        self,
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
    
@hydra.main(config_path=f"{DIR_PATH}/configs", config_name="fingereye_lab", version_base="1.1")
def main(config):
    cprint(OmegaConf.to_yaml(config), "grey")
    """Zero actions agent with Isaac Lab environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # check if randomization is enabled
    if args_cli.no_rand_all:
        env_cfg.randomization.enable_all = False
    if args_cli.no_rand_light:
        env_cfg.randomization.random_lighting = False
    if args_cli.no_rand_coin:
        env_cfg.randomization.random_object_color = False
    if args_cli.no_rand_camera:
        env_cfg.randomization.random_camera_noise = False
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    env.reset(seed=env_cfg.seed)
    recorder = SimulatorRecorder(env=env.unwrapped, config=config)
    if config.mode == "record":
        recorder.run_record()
    else:
        # recorder.run_policy_multi_seeds(seeds=[0, 1, 2])
        recorder.run_policy()

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    simulation_app.close()
