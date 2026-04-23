import os
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_visual_material, get_current_stage
from isaaclab.utils.math import quat_apply, sample_uniform, matrix_from_quat
import torch.nn.functional as F
from .fingereye_env_cfg import FingerEyeLabEnvCfg

FILE_PATH = Path(__file__).resolve()
DIR_PATH = FILE_PATH.parents[6]

from .env_tools import (
    create_sky_sphere_mesh,
    create_skybox_material_from_hdri,
    ensure_floor_uv,
    augment_image_batch,
    quat_to_M6,
)

class FingerEyeLabEnv(DirectRLEnv):
    cfg: FingerEyeLabEnvCfg

    def __init__(self, cfg: FingerEyeLabEnvCfg, render_mode: str | None = None, **kwargs):
        self.rand_cfg = cfg.randomization
        self._init_skybox_assets()
        super().__init__(cfg, render_mode, **kwargs)
        print("\n" + "=" * 80)
        print("DEBUG: Verifying Joint Actuator Parameters")
        print(f"{'Idx':<5} | {'Joint Name':<35} | {'Stiffness':<10} | {'Damping':<10} | {'Lower Limit':<10} | {'Upper Limit':<10}")
        print("-" * 80)
        # Access default parameters from env 0 (assuming they are uniform across envs)
        # These tensors are on the device (GPU), so we use .item() to read them
        stiff_data = self.hand.data.default_joint_stiffness[0]
        damp_data = self.hand.data.default_joint_damping[0]
        lower_joint_limits = self.hand.data.joint_pos_limits[0, :, 0]
        upper_joint_limits = self.hand.data.joint_pos_limits[0, :, 1]
        
        for i, name in enumerate(self.hand.joint_names):
            k = stiff_data[i].item()
            d = damp_data[i].item()
            lower_limit = lower_joint_limits[i].item()
            upper_limit = upper_joint_limits[i].item()
            print(f"{i:<5} | {name:<35} | {k:<10.1f} | {d:<10.1f} | {lower_limit:<10.3f} | {upper_limit:<10.3f}")
            
        print("=" * 80 + "\n")
        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.control_dof_indices = list()
        for joint_name in cfg.control_joint_names:
            self.control_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices = list()
        for joint_name in cfg.all_actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.init_joint_values = torch.zeros((self.num_hand_dofs,), dtype=torch.float, device=self.device)
        viser_init_joint_values = torch.concat(
            [
                torch.deg2rad(torch.tensor(cfg.xarm_init_joint_values_degree, device=self.device)),
                torch.tensor(cfg.leap_init_joint_values_radian, device=self.device) - np.pi
            ]
        )
        self.init_joint_values[self.actuated_dof_indices] = viser_init_joint_values

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        
        self.local_normal_vector = torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1)

        # -----------------------------------------------------------------------
        # Tag / Holder Joint Index Resolution
        # -----------------------------------------------------------------------
        # We look for the 6 degrees of freedom (DOF) associated with each tag holder.
        # Typically named: prefix_px, prefix_py, prefix_pz, prefix_rx, prefix_ry, prefix_rz
        
        dof_suffixes = ["_px", "_py", "_pz", "_rx", "_ry", "_rz"]
        collected_tag_indices = []

        # Assuming cfg.enabled_tag_joints_prefix is the list ["fingertip_holder", "thumb_holder"]
        if hasattr(cfg, "enabled_tag_joints_prefix") and cfg.enabled_tag_joints_prefix:
            for prefix in cfg.enabled_tag_joints_prefix:
                indices = []
                for suffix in dof_suffixes:
                    joint_name = f"{prefix}{suffix}"
                    
                    if joint_name in self.hand.joint_names:
                        indices.append(self.hand.joint_names.index(joint_name))
                    else:
                        print(f"[WARNING] FingerEyeLabEnv: Tag joint '{joint_name}' not found in articulation.")
                
                # Ensure we found all 6 DOFs for this tag
                if len(indices) == 6:
                    collected_tag_indices.append(indices)
                else:
                    print(f"[ERROR] FingerEyeLabEnv: Incomplete DOFs for tag '{prefix}'. Found {len(indices)}/6.")

        # self.tag_joint_indices: shape = (n_tags, 6)
        # Columns correspond to [px, py, pz, rx, ry, rz] indices in the full q vector
        self.tag_joint_indices = torch.tensor(collected_tag_indices, dtype=torch.long, device=self.device)


        # 2 finger tip links 
        self.fingertip_link_names = ["fingertip_holder", "thumb_soft_ring"]
        self.fingertip_indices = [
            i for i, name in enumerate(self.hand.body_names)  if name in self.fingertip_link_names
        ]
        # Fallback if no specific fingers found (use all bodies as a safe default)
        if len(self.fingertip_indices) == 0:
            self.fingertip_indices = list(range(self.hand.num_bodies))
        
        # Convert to tensor for efficient indexing
        self.fingertip_indices = torch.tensor(self.fingertip_indices, dtype=torch.long, device=self.device)

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.setup_gray_ground()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        # UV Generation
        stage = omni.usd.get_context().get_stage()
        for env_id in range(self.num_envs):
            floor_prim_path = f"/World/envs/env_{env_id}/Background/floor"
            ensure_floor_uv(stage, floor_prim_path)
        # ------------------------------------------------
        if self.rand_cfg.enable_all or self.rand_cfg.random_lighting:
            self._setup_env_lights()
        else:
            light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75), visible_in_primary_ray=self.rand_cfg.visible_in_primary_ray)
            light_cfg.func("/World/Light", light_cfg)
        if self.rand_cfg.enable_all or self.rand_cfg.random_background:
            self._setup_env_backgrounds()
            for env_id in range(self.num_envs):
                self._setup_env_skybox_and_light_for_env(env_id)


        # cameras 
        if self.cfg.enable_cameras:
            self._camera_configs = {
                "wrist_camera": self.cfg.cam_wrist if hasattr(self.cfg, 'cam_wrist') else None,
                "index_tip":  self.cfg.cam_index_tip if hasattr(self.cfg, 'cam_index_tip') else None,
                "index_root": self.cfg.cam_index_root if hasattr(self.cfg, 'cam_index_root') else None,
                "thumb_tip":  self.cfg.cam_thumb_tip if hasattr(self.cfg, 'cam_thumb_tip') else None,
                "thumb_root": self.cfg.cam_thumb_root if hasattr(self.cfg, 'cam_thumb_root') else None,
                "third_view": self.cfg.cam_third_view if hasattr(self.cfg, 'cam_third_view') else None,
            }

            self._active_cameras = {} 

            # Initialize requested cameras
            for name in self.cfg.camera_name_list:
                if name in self._camera_configs:
                    print(f"[INFO] Spawning Camera: {name} ({self.cfg.img_w}x{self.cfg.img_h})")
                    sensor = TiledCamera(self._camera_configs[name])
                    self.scene.sensors[name] = sensor
                    self._active_cameras[name] = sensor

    def _pre_physics_step(self, actions: torch.Tensor|dict) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        if self.cfg.replay_mode:
            # replay mode, input (ne, num_dof + 3 + 4) for action, pos and rot
            # robot action
            current_joints = self.actions[:, :self.num_hand_dofs] # (ne, num_dof)
            current_velocities = torch.zeros_like(current_joints)
            self.hand.write_joint_state_to_sim(position=current_joints, velocity=current_velocities)
            # coin action
            current_coin_pose = self.actions[:, self.num_hand_dofs:] # (ne, 7)
            current_coin_pose[:, :3] += self.scene.env_origins # (ne, 3) # + self.scene.env_origins because we collected in local frame way
            self.object.write_root_pose_to_sim(current_coin_pose)
            self.object.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device))
        else:
            self.cur_targets[:] = self.init_joint_values.clone()
            self.cur_targets[:, self.control_dof_indices] = self.actions
            self.hand.set_joint_position_target(self.cur_targets)

    def _get_observations(self) -> dict:
        obs = {}
        self._compute_intermediate_values()

        # -----------------------------------------------------------------------
        # Camera Handling
        # -----------------------------------------------------------------------
        if self.cfg.enable_cameras:
            rs_image_list = []
            rgb_image_list = []
            third_view_image = None 
            
            # Pose lists (Stores 9D vectors: 3 pos + 6 rot)
            rs_pose_list = []
            rgb_pose_list = []
            third_view_pose = None

            for name in self.cfg.camera_name_list:
                if name in self._active_cameras:
                    cam_sensor = self._active_cameras[name]
                    
                    cam_data = cam_sensor.data.output["rgb"]
                
                    pos_env = cam_sensor.data.pos_w - self.scene.env_origins
                    rot_mat = matrix_from_quat(cam_sensor.data.quat_w_world)
                    rot_9d = rot_mat.reshape(self.num_envs, -1)
                    
                    cam_pose = torch.cat([pos_env, rot_9d], dim=-1)

                    if name in ["wrist_camera"]:
                        rs_image_list.append(cam_data.unsqueeze(1)) 
                        rs_pose_list.append(cam_pose.unsqueeze(1)) 
                    elif name == "third_view":
                        third_view_image = cam_data
                        third_view_pose = cam_pose
                    else:
                        rgb_image_list.append(cam_data.unsqueeze(1))
                        rgb_pose_list.append(cam_pose.unsqueeze(1))

            # Concatenate Results
            rs_images = torch.cat(rs_image_list, dim=1) if len(rs_image_list) > 0 else None
            rgb_images = torch.cat(rgb_image_list, dim=1) if len(rgb_image_list) > 0 else None
            
            rs_poses = torch.cat(rs_pose_list, dim=1) if len(rs_pose_list) > 0 else None
            rgb_poses = torch.cat(rgb_pose_list, dim=1) if len(rgb_pose_list) > 0 else None

            if self.rand_cfg.enable_all or self.rand_cfg.random_camera_noise:
                rs_images = augment_image_batch(rs_images)
                rgb_images = augment_image_batch(rgb_images)
                
            # Populate Obs
            obs["third_view"] = third_view_image             # (N, H, W, 3)
            obs["third_view_pose"] = third_view_pose         # (N, 9)
            
            obs["current_rs_images"] = rs_images             # (N, n_cams, H, W, 3)
            obs["current_rs_camera_poses"] = rs_poses        # (N, n_cams, 12)
            
            obs["current_rgb_images"] = rgb_images           # (N, n_cams, H, W, 3)
            obs["current_rgb_camera_poses"] = rgb_poses      # (N, n_cams, 12)
            
        else:
            obs["third_view"] = None
            obs["third_view_pose"] = None
            obs["current_rs_images"] = None
            obs["current_rs_camera_poses"] = None
            obs["current_rgb_images"] = None
            obs["current_rgb_camera_poses"] = None

        # -----------------------------------------------------------------------
        # Joint & Object States
        # -----------------------------------------------------------------------
        obs["current_joint_values"] = self.hand_dof_pos[:, self.control_dof_indices]
        obs["current_all_joint_values"] = self.hand_dof_pos.clone()
        obs["coin_pose"] = torch.cat((self.object_pos, self.object_rot), dim=-1)
        obs["coin_z_axis"] = quat_to_M6(self.object_rot)
        obs["pos_of_coin"] = self.object_pos
        # -----------------------------------------------------------------------
        # Tag Poses
        # -----------------------------------------------------------------------
        if self.tag_joint_indices.numel() > 0:
            obs["current_center_tag_T"] = self.hand_dof_pos[:, self.tag_joint_indices]
        else:
            obs["current_center_tag_T"] = torch.empty((self.num_envs, 0, 6), device=self.device)

        return obs

    def _get_rewards(self) -> torch.Tensor:
        # self.coin_normal_w is refreshed in _compute_intermediate_values
        normal_z_abs = torch.abs(self.coin_normal_w[:, 2])
        angle_tolerance = 10 / 180 * np.pi # 5 degrees in radians
        sin_angle_tolerance = np.sin(angle_tolerance) # 0.17453

        # Reward Return Home: Only if success
        fingertip_pos = self.hand.data.body_pos_w[:, self.fingertip_indices]  # (ne, n_fingertips, 3)
        first_fingertip_pos = fingertip_pos[:, 0, :]  # (ne, 3)
        second_fingertip_pos = fingertip_pos[:, 1, :]  # (ne, 3)
        dist_between_fingertips = torch.norm(first_fingertip_pos - second_fingertip_pos, dim=-1)   # (ne,)

        # Success Condition
        is_standing = (normal_z_abs < sin_angle_tolerance) & (dist_between_fingertips > 0.12)
        is_standing_float = is_standing.float()
        reward = is_standing_float
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # NOTE: simplify, never done, only success / not  
        return False, False

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        random_env_ids = env_ids[1:]
        # randomizations
        if self.rand_cfg.enable_all or self.rand_cfg.random_lighting:
            self._randomize_env_lighting(random_env_ids)
        if self.rand_cfg.enable_all or self.rand_cfg.random_coin_color:
            self._randomize_coin_mdl_color(random_env_ids)
        if self.rand_cfg.enable_all or self.rand_cfg.random_background:
            self._randomize_env_skybox(env_ids) # do not need randomize skybox
            self._randomize_env_floor_color(random_env_ids, base_gray=0.7, delta=0.3)
        # reset object
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_x = sample_uniform(
            self.cfg.coin_x_min, self.cfg.coin_x_max, (len(env_ids),), device=self.device
        )
        object_y = sample_uniform(
            self.cfg.coin_y_min, self.cfg.coin_y_max, (len(env_ids),), device=self.device
        )
        object_default_state[:, 0] = object_x
        object_default_state[:, 1] = object_y
        object_default_state[:, 2] = self.cfg.coin_z
        object_default_state[:, 0:3] = (
            object_default_state[:, 0:3] + self.scene.env_origins[env_ids]
        )
        object_default_state[:, 7:] = torch.zeros_like(self.object.data.default_root_state[env_ids, 7:])
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)
        print(f"[object pos]:{self.object.data.root_pos_w - self.scene.env_origins}")

        # reset hand
        dof_pos = self.init_joint_values.unsqueeze(0).repeat(len(env_ids), 1)
        dof_vel = self.hand.data.default_joint_vel[env_ids] 
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        # Initialize obs/buffers immediately
        self._compute_intermediate_values(env_ids)
        self.sim.step()


    def _compute_intermediate_values(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
            
        self.hand_dof_pos = self.hand.data.joint_pos
        self.actuated_dof_pos = self.hand_dof_pos[:, self.actuated_dof_indices]
        self.hand_dof_vel = self.hand.data.joint_vel

        # data for object
        self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
        self.object_rot = self.object.data.root_quat_w
        self.object_velocities = self.object.data.root_vel_w
        self.object_linvel = self.object.data.root_lin_vel_w
        self.object_angvel = self.object.data.root_ang_vel_w

        # --- Calculate Coin Face Normal (World) ---
        # Rot * Local_Y
        self.coin_normal_w = quat_apply(self.object_rot, self.local_normal_vector)

    def replay_step(self, action: torch.Tensor):
        """Execute one time-step of the environment's dynamics.

        The environment steps forward at a fixed time-step, while the physics simulation is decimated at a
        lower time-step. This is to ensure that the simulation is stable. These two time-steps can be configured
        independently using the :attr:`DirectRLEnvCfg.decimation` (number of simulation steps per environment step)
        and the :attr:`DirectRLEnvCfg.sim.physics_dt` (physics time-step). Based on these parameters, the environment
        time-step is computed as the product of the two.

        This function performs the following steps:

        1. Pre-process the actions before stepping through the physics.
        2. Apply the actions to the simulator and step through the physics in a decimated manner.
        3. Compute the reward and done signals.
        4. Reset environments that have terminated or reached the maximum episode length.
        5. Apply interval events if they are enabled.
        6. Compute observations.

        Args:
            action: The actions to apply on the environment. Shape is (num_envs, action_dim).

        Returns:
            A tuple containing the observations, rewards, resets (terminated and truncated) and extras.
        """
        action = action.to(self.device)
        # add action noise
        if self.cfg.action_noise_model:
            action = self._action_noise_model(action)

        # process actions
        self._pre_physics_step(action)

        # check if we need to do rendering within the physics loop
        # note: checked here once to avoid multiple checks within the loop
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()

        # perform physics stepping
        # set actions into buffers
        self._apply_action()
        # set actions into simulator
        self.scene.write_data_to_sim()
        # simulate
        # self.sim.step(render=False)
        # render between steps only if the GUI or an RTX sensor needs it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        self.sim.render()
        # update buffers at sim dt
        self.scene.update(dt=0)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()

        # -- reset envs that terminated/timed-out and log the episode information
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
            # if sensors are added to the scene, make sure we render to reflect changes in reset
            if self.sim.has_rtx_sensors() and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

        # post-step: step interval event
        if self.cfg.events:
            if "interval" in self.event_manager.available_modes:
                self.event_manager.apply(mode="interval", dt=self.step_dt)

        # update observations
        self.obs_buf = self._get_observations()

        # add observation noise
        # note: we apply no noise to the state space (since it is used for critic networks)
        if self.cfg.observation_noise_model:
            self.obs_buf["policy"] = self._observation_noise_model(self.obs_buf["policy"])

        # return observations, rewards, resets and extras
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _init_skybox_assets(self):
        skybox_root = Path(DIR_PATH) / "assets" / "skybox"

        folders = []
        hdr_files = []

        for sub in sorted(skybox_root.iterdir()):
            if not sub.is_dir():
                continue
            candidates = list(sub.glob("*_HDR.exr"))
            if not candidates:
                continue
            hdr = candidates[0]
            folders.append(sub.name)
            hdr_files.append(str(hdr))

        self.skybox_folders = folders
        self.skybox_hdr_files = hdr_files

    def _randomize_coin_mdl_color(self, env_ids=None):
        stage = omni.usd.get_context().get_stage()

        if env_ids is None:
            env_ids = range(self.num_envs)

        for i in env_ids:
            r = random.random()   
            g = random.random()
            b = random.random()
            color = Gf.Vec3f(r, g, b)
            shader_path = f"/World/envs/env_{i}/Coin/material/Shader"
            prim = stage.GetPrimAtPath(shader_path)
            if not prim.IsValid():
                continue

            attr_name = "inputs:diffuseColor"
            attr = prim.GetAttribute(attr_name)
            attr.Set(color)

    def _setup_env_lights(self):

        stage = omni.usd.get_context().get_stage()

        self.env_light_paths = []

        for env_id in range(self.num_envs):
            env_prim_path = f"/World/envs/env_{env_id}"
            light_path = f"{env_prim_path}/EnvLight"
            self.env_light_paths.append(light_path)

            if stage.GetPrimAtPath(light_path).IsValid():
                continue

            light = UsdLux.RectLight.Define(stage, light_path)

            xformable = UsdGeom.Xformable(light.GetPrim())
            xformable.AddTranslateOp().Set(Gf.Vec3f(0.4, 0.11, 2.4))
            xformable.AddRotateXYZOp().Set(Gf.Vec3f(-10.0, 0.0, 0.0))

            w_attr = light.GetWidthAttr()
            if not w_attr:
                w_attr = light.CreateWidthAttr()
            w_attr.Set(0.3)

            h_attr = light.GetHeightAttr()
            if not h_attr:
                h_attr = light.CreateHeightAttr()
            h_attr.Set(0.3)

            inten_attr = light.GetIntensityAttr()
            if not inten_attr:
                inten_attr = light.CreateIntensityAttr()
            inten_attr.Set(200000.0)

            color_attr = light.GetColorAttr()
            if not color_attr:
                color_attr = light.CreateColorAttr()
            color_attr.Set(Gf.Vec3f(1.0, 1.0, 1.0))

    def _setup_env_backgrounds(self):

        stage = omni.usd.get_context().get_stage()
        self.env_bg_root_paths = []

        room_half_width = 1.0  
        room_half_depth = 1.0  

        def make_floor_mesh(panel_path: str, translate: Gf.Vec3f, size_x: float, size_y: float):
            prim = stage.GetPrimAtPath(panel_path)
            if prim.IsValid():
                if prim.GetTypeName() == "Mesh":
                    return
                stage.RemovePrim(panel_path)

            mesh = UsdGeom.Mesh.Define(stage, panel_path)

            points = [
                Gf.Vec3f(-0.5, -0.5, 0.0),
                Gf.Vec3f( 0.5, -0.5, 0.0),
                Gf.Vec3f( 0.5,  0.5, 0.0),
                Gf.Vec3f(-0.5,  0.5, 0.0),
            ]
            mesh.CreatePointsAttr(points)
            mesh.CreateFaceVertexCountsAttr([4])
            mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])

            mesh.CreateNormalsAttr(
                [Gf.Vec3f(0.0, 0.0, 1.0)] * 4
            )
            mesh.SetNormalsInterpolation("vertex")
            primvars_api = UsdGeom.PrimvarsAPI(mesh)
            st = primvars_api.CreatePrimvar(
                "st",
                Sdf.ValueTypeNames.TexCoord2fArray,
                UsdGeom.Tokens.varying,
            )
            st.Set(
                [
                    Gf.Vec2f(0, 0.0),
                    Gf.Vec2f(1.0, 0.0),
                    Gf.Vec2f(1.0, 1.0),
                    Gf.Vec2f(0, 1.0),
                ]
            )

            xform = UsdGeom.Xformable(mesh.GetPrim())
            xform.AddTranslateOp().Set(translate)
            xform.AddScaleOp().Set(Gf.Vec3f(2.0 * room_half_width, 2.0 * room_half_depth, 1.0))
            mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.7, 0.7, 0.7)])

        for env_id in range(self.num_envs):
            env_root = f"/World/envs/env_{env_id}"
            bg_root = f"{env_root}/Background"
            self.env_bg_root_paths.append(bg_root)

            make_floor_mesh(
                panel_path=f"{bg_root}/floor",
                translate=Gf.Vec3f(0.0, 0.0, 0.0001),
                size_x=2.0 * room_half_width,
                size_y=2.0 * room_half_depth,
            )              

    def _randomize_env_lighting(
        self,
        env_ids=None,
        intensity_range=(2e5, 5e6),   
        white_jitter=0.05,            
    ):
        stage = omni.usd.get_context().get_stage()

        if env_ids is None:
            env_ids = range(self.num_envs)

        for env_id in env_ids:
            if env_id < 0 or env_id >= self.num_envs:
                continue

            if hasattr(self, "env_light_paths"):
                light_path = self.env_light_paths[env_id]
            else:
                light_path = f"/World/envs/env_{env_id}/EnvLight"

            prim = stage.GetPrimAtPath(light_path)
            if not prim.IsValid():
                continue

            light = UsdLux.RectLight(prim)

            min_intensity, max_intensity = intensity_range
            intensity = float(
                torch.empty(1, device=self.device).uniform_(min_intensity, max_intensity).item()
            )
            inten_attr = light.GetIntensityAttr()
            if not inten_attr:
                inten_attr = light.CreateIntensityAttr()
            inten_attr.Set(intensity)

            color_attr = light.GetColorAttr()
            if not color_attr:
                color_attr = light.CreateColorAttr()

            if white_jitter <= 0:
                # Use pure white when no jitter is requested.
                color = (1.0, 1.0, 1.0)
            else:
                base_color = torch.ones(3, device=self.device)
                color_tensor = base_color + torch.empty(3, device=self.device).uniform_(
                    -white_jitter, white_jitter
                )
                color_tensor = torch.clamp(color_tensor, 0.0, 1.0)
                color = tuple(color_tensor.tolist())

            color_attr.Set(Gf.Vec3f(*color))

    def _randomize_env_floor_color(self, env_ids=None, base_gray: float = 0.7, delta: float = 0.1):

        if env_ids is None:
            env_ids = range(self.num_envs)

        stage = omni.usd.get_context().get_stage()

        for env_id in env_ids:
            floor_path = f"/World/envs/env_{env_id}/Background/floor"
            floor_prim = stage.GetPrimAtPath(floor_path)

            if not floor_prim.IsValid():
                continue

            rand = torch.rand(1, device=self.device).item()  # 0 ~ 1
            gray = base_gray + (2.0 * rand - 1.0) * delta    
            gray = max(0.0, min(1.0, float(gray)))
            color = Gf.Vec3f(gray, gray, gray)

            gprim = UsdGeom.Gprim(floor_prim)
            display_color_attr = gprim.GetDisplayColorAttr()
            if not display_color_attr:
                display_color_attr = gprim.CreateDisplayColorAttr()

            display_color_attr.Set([color])

    def _setup_env_skybox_and_light_for_env(self, env_id: int):

        stage = omni.usd.get_context().get_stage()
        env_root = f"/World/envs/env_{env_id}"

        sky_root = Path(DIR_PATH) / "assets" / "skybox"
        folder = self.skybox_folders[env_id % len(self.skybox_folders)]
        tex_dir = sky_root / folder
        files = os.listdir(tex_dir)
        tonemapped = [f for f in files if f.endswith("_TONEMAPPED.jpg")]
        if tonemapped:
            tex_file = tonemapped[0]
        else:
            candidates = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            tex_file = candidates[0]
        tex_path = tex_dir / tex_file

        dome_path = f"{env_root}/SkyDome"
        sky_mesh = create_sky_sphere_mesh(stage, dome_path, radius=5.0)
    
        mat_prim = f"/World/Looks/skybox_{folder}"
        material = create_skybox_material_from_hdri(stage, mat_prim, str(tex_dir))
        if material:
            UsdShade.MaterialBindingAPI(sky_mesh.GetPrim()).Bind(material)
        else:
            gprim = UsdGeom.Gprim(sky_mesh.GetPrim())
            gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(0.2, 0.2, 0.25)])

    def _randomize_env_skybox(self, env_ids=None):

        if env_ids is None:
            env_ids = range(self.num_envs)

        stage = omni.usd.get_context().get_stage()
        num_skyboxes = len(self.skybox_folders)
        sky_root = Path(DIR_PATH) / "assets" / "skybox"

        for env_id in env_ids:
            if env_id < 0 or env_id >= self.num_envs:
                continue

            idx_tensor = torch.randint(0, num_skyboxes, (1,), device=self.device)
            sky_idx = int(idx_tensor.item())

            folder = self.skybox_folders[sky_idx]
            tex_dir = sky_root / folder

            files = os.listdir(tex_dir)
            tonemapped = [f for f in files if f.endswith("_TONEMAPPED.jpg")]
            if tonemapped:
                tex_file = tonemapped[0]
            else:
                candidates = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png", ".exr", ".hdr"))]

                tex_file = candidates[0]

            tex_path = str(tex_dir / tex_file)

            dome_path = f"/World/envs/env_{env_id}/SkyDome"
            dome_prim = stage.GetPrimAtPath(dome_path)
            dome = UsdLux.DomeLight(dome_prim)

            tex_attr = dome.GetTextureFileAttr()
            if not tex_attr:
                tex_attr = dome.CreateTextureFileAttr()
            tex_attr.Set(Sdf.AssetPath(tex_path))

            fmt_attr = dome.GetTextureFormatAttr()
            if not fmt_attr:
                fmt_attr = dome.CreateTextureFormatAttr()
            fmt_attr.Set("latlong")

    def setup_gray_ground(self):
        stage = get_current_stage()
        ground_path = "/World/ground"
        material_path = f"{ground_path}/Looks/OverwriteGray"

        # Define a new Material and Shader
        mat_prim = UsdShade.Material.Define(stage, material_path)
        shader_prim = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader_prim.CreateIdAttr("UsdPreviewSurface")

        # Set the Color to (0.3, 0.3, 0.3)
        shader_prim.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.3, 0.3, 0.3))
        shader_prim.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0) # Matte
        shader_prim.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        # Connect Shader -> Material
        mat_prim.CreateSurfaceOutput().ConnectToSource(shader_prim.ConnectableAPI(), "surface")

        # Force bind this material to the ground, overriding the grid
        bind_visual_material(ground_path, material_path)

