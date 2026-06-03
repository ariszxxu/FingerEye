import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_visual_material, get_current_stage
from isaaclab.utils.math import quat_apply, matrix_from_quat
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


def _quat_wxyz_to_matrix_np(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-9)
    w, x, y, z = quat.tolist()
    return np.asarray(
        [
            [w * w + x * x - y * y - z * z, 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), w * w - x * x + y * y - z * z, 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), w * w - x * x - y * y + z * z],
        ],
        dtype=np.float32,
    )


def _gf_quat_to_wxyz_np(quat) -> np.ndarray:
    imag = quat.GetImaginary()
    return np.asarray([quat.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float32)


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
        collected_tag_anchor_body_indices = []

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
                    anchor_body_name = prefix
                    # Isaac may collapse zero-offset fixed links. thumb_holder is
                    # fixed to thumb_fingertip in the URDF, so thumb_fingertip is
                    # the correct world-frame anchor when thumb_holder is absent.
                    if anchor_body_name not in self.hand.body_names and prefix == "thumb_holder":
                        anchor_body_name = "thumb_fingertip"
                    if anchor_body_name in self.hand.body_names:
                        collected_tag_anchor_body_indices.append(self.hand.body_names.index(anchor_body_name))
                    else:
                        collected_tag_anchor_body_indices.append(-1)
                        print(f"[WARNING] FingerEyeLabEnv: Tag anchor body '{prefix}' not found.")
                else:
                    print(f"[ERROR] FingerEyeLabEnv: Incomplete DOFs for tag '{prefix}'. Found {len(indices)}/6.")

        # self.tag_joint_indices: shape = (n_tags, 6)
        # Columns correspond to [px, py, pz, rx, ry, rz] indices in the full q vector
        self.tag_joint_indices = torch.tensor(collected_tag_indices, dtype=torch.long, device=self.device)
        self.tag_anchor_body_indices = torch.tensor(
            collected_tag_anchor_body_indices, dtype=torch.long, device=self.device
        )


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
        self.surface_fingertip_link_names = list(getattr(self.cfg, "contact_fingertip_link_names", []))
        surface_indices = []
        for name in self.surface_fingertip_link_names:
            if name in self.hand.body_names:
                surface_indices.append(self.hand.body_names.index(name))
            else:
                print(f"[WARNING] Fingertip surface link '{name}' not found in articulation bodies.")
        self.contact_fingertip_indices = torch.tensor(surface_indices, dtype=torch.long, device=self.device)

    def _clone_camera_with_standard_xform_ops(self, stage, camera_cfg):
        for prim in sim_utils.find_matching_prims(camera_cfg.prim_path, stage=stage):
            prim_path = prim.GetPath().pathString
            source_xformable = UsdGeom.Xformable(prim)
            transform = Gf.Transform(source_xformable.GetLocalTransformation())
            translation = Gf.Vec3d(transform.GetTranslation())
            orientation = Gf.Quatd(transform.GetRotation().GetQuat())
            scale = Gf.Vec3d(transform.GetScale())
            reset_stack = source_xformable.GetResetXformStack()

            attrs_to_copy = []
            for attr in prim.GetAttributes():
                attr_name = attr.GetName()
                if attr_name.startswith("xformOp:") or attr_name == "xformOpOrder":
                    continue
                attr_value = attr.Get()
                if attr_value is None:
                    continue
                attrs_to_copy.append((attr_name, attr.GetTypeName(), attr.IsCustom(), attr_value))

            stage.RemovePrim(prim_path)
            camera = UsdGeom.Camera.Define(stage, prim_path)
            clean_prim = camera.GetPrim()
            for attr_name, type_name, is_custom, attr_value in attrs_to_copy:
                clean_attr = clean_prim.GetAttribute(attr_name)
                if not clean_attr:
                    clean_attr = clean_prim.CreateAttribute(attr_name, type_name, custom=is_custom)
                clean_attr.Set(attr_value)

            self._set_camera_xform_ops(clean_prim, translation, orientation, scale, reset_stack)

    def _set_camera_xform_ops(self, prim, translation, orientation, scale, reset_stack=False):
        xformable = UsdGeom.Xformable(prim)
        with Sdf.ChangeBlock():
            for prop_name in prim.GetPropertyNames():
                if prop_name.startswith("xformOp:") or prop_name == "xformOpOrder":
                    prim.RemoveProperty(prop_name)

            xformable.SetXformOpOrder([], reset_stack)

            translate_op = xformable.AddXformOp(UsdGeom.XformOp.TypeTranslate, UsdGeom.XformOp.PrecisionDouble, "")
            orient_op = xformable.AddXformOp(UsdGeom.XformOp.TypeOrient, UsdGeom.XformOp.PrecisionDouble, "")
            scale_op = xformable.AddXformOp(UsdGeom.XformOp.TypeScale, UsdGeom.XformOp.PrecisionDouble, "")
            translate_op.Set(translation)
            orient_op.Set(orientation)
            scale_op.Set(scale)
            xformable.SetXformOpOrder([translate_op, orient_op, scale_op], reset_stack)

    def _standardize_camera_xform_ops(self, prim, source_prim=None):
        xformable = UsdGeom.Xformable(prim)
        source_xformable = UsdGeom.Xformable(source_prim) if source_prim is not None else xformable
        transform = Gf.Transform(source_xformable.GetLocalTransformation())
        translation = Gf.Vec3d(transform.GetTranslation())
        orientation = Gf.Quatd(transform.GetRotation().GetQuat())
        scale = Gf.Vec3d(transform.GetScale())
        reset_stack = source_xformable.GetResetXformStack()
        self._set_camera_xform_ops(prim, translation, orientation, scale, reset_stack)

    def _camera_parent_body_name(self, prim_path: str) -> str | None:
        parts = prim_path.replace(".*", "0").split("/")
        if "xarm7" not in parts:
            return None
        idx = parts.index("xarm7")
        if idx + 2 >= len(parts):
            return None
        return parts[-2]

    def _camera_local_pose_from_cfg_or_stage(self, stage, camera_cfg):
        if getattr(camera_cfg, "offset", None) is not None:
            pos = np.asarray(camera_cfg.offset.pos, dtype=np.float32)
            rot = _quat_wxyz_to_matrix_np(np.asarray(camera_cfg.offset.rot, dtype=np.float32))
            return pos, rot

        prims = list(sim_utils.find_matching_prims(camera_cfg.prim_path, stage=stage))
        if len(prims) == 0:
            return None, None
        transform = Gf.Transform(UsdGeom.Xformable(prims[0]).GetLocalTransformation())
        pos = np.asarray(transform.GetTranslation(), dtype=np.float32)
        rot = _quat_wxyz_to_matrix_np(_gf_quat_to_wxyz_np(transform.GetRotation().GetQuat()))
        return pos, rot

    def _register_camera_body_pose_source(self, name: str, camera_cfg, stage) -> None:
        body_name = self._camera_parent_body_name(camera_cfg.prim_path)
        if body_name is None:
            return
        local_pos, local_rot = self._camera_local_pose_from_cfg_or_stage(stage, camera_cfg)
        if local_pos is None or local_rot is None:
            return
        self._camera_body_names[name] = body_name
        self._camera_local_pos[name] = torch.as_tensor(local_pos, dtype=torch.float32, device=self.device)
        self._camera_local_rot[name] = torch.as_tensor(local_rot, dtype=torch.float32, device=self.device)

    def _camera_pose_from_articulation(self, name: str, sensor_pose=None):
        if name not in self._camera_body_indices:
            body_name = self._camera_body_names.get(name)
            if body_name is None or body_name not in self.hand.body_names:
                return None
            self._camera_body_indices[name] = self.hand.body_names.index(body_name)
        body_idx = self._camera_body_indices[name]
        body_pos_env = self.hand.data.body_pos_w[:, body_idx] - self.scene.env_origins
        body_rot = matrix_from_quat(self.hand.data.body_quat_w[:, body_idx])
        if name not in self._camera_runtime_local_pos and sensor_pose is not None:
            sensor_pos = sensor_pose[:, :3].to(device=self.device, dtype=body_pos_env.dtype)
            sensor_rot = sensor_pose[:, 3:].reshape(self.num_envs, 3, 3).to(device=self.device, dtype=body_rot.dtype)
            body_rot_t = body_rot.transpose(1, 2)
            self._camera_runtime_local_pos[name] = torch.bmm(
                body_rot_t,
                (sensor_pos - body_pos_env).unsqueeze(-1),
            ).squeeze(-1).detach()
            self._camera_runtime_local_rot[name] = torch.bmm(body_rot_t, sensor_rot).detach()

        if name in self._camera_runtime_local_pos:
            local_pos = self._camera_runtime_local_pos[name].to(device=self.device, dtype=body_pos_env.dtype)
            local_rot = self._camera_runtime_local_rot[name].to(device=self.device, dtype=body_rot.dtype)
        else:
            local_pos = self._camera_local_pos[name].to(device=self.device, dtype=body_pos_env.dtype).view(1, 3)
            local_rot = self._camera_local_rot[name].to(device=self.device, dtype=body_rot.dtype).view(1, 3, 3)
            local_pos = local_pos.expand(self.num_envs, -1)
            local_rot = local_rot.expand(self.num_envs, -1, -1)
        cam_pos = body_pos_env + torch.bmm(body_rot, local_pos.unsqueeze(-1)).squeeze(-1)
        cam_rot = torch.bmm(body_rot, local_rot)
        return torch.cat([cam_pos, cam_rot.reshape(self.num_envs, -1)], dim=-1)

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        object_cfg = getattr(self.cfg, "object_cfg", None)
        self.object = RigidObject(object_cfg) if object_cfg is not None else None
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.setup_gray_ground()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        if self.object is not None:
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
            self._camera_body_names = {}
            self._camera_body_indices = {}
            self._camera_local_pos = {}
            self._camera_local_rot = {}
            self._camera_runtime_local_pos = {}
            self._camera_runtime_local_rot = {}

            # Initialize requested cameras
            for name in self.cfg.camera_name_list:
                if name in self._camera_configs and self._camera_configs[name] is not None:
                    camera_cfg = self._camera_configs[name]
                    print(f"[INFO] Spawning Camera: {name} ({camera_cfg.width}x{camera_cfg.height})")
                    if camera_cfg.spawn is None:
                        self._clone_camera_with_standard_xform_ops(stage, camera_cfg)
                    sensor = TiledCamera(camera_cfg)
                    self.scene.sensors[name] = sensor
                    self._active_cameras[name] = sensor
                    self._register_camera_body_pose_source(name, camera_cfg, stage)

    def _pre_physics_step(self, actions: torch.Tensor|dict) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        if self.cfg.replay_mode:
            current_joints = self.actions[:, :self.num_hand_dofs] # (ne, num_dof)
            current_velocities = torch.zeros_like(current_joints)
            self.hand.write_joint_state_to_sim(position=current_joints, velocity=current_velocities)
            if self.object is not None and self.actions.shape[1] >= self.num_hand_dofs + 7:
                object_pose = self.actions[:, self.num_hand_dofs : self.num_hand_dofs + 7].clone()
                object_pose[:, :3] += self.scene.env_origins
                self.object.write_root_pose_to_sim(object_pose)
                self.object.write_root_velocity_to_sim(
                    torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device)
                )
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
                
                    pos_env = cam_sensor.data.pos_w - self.scene.env_origins
                    rot_mat = matrix_from_quat(cam_sensor.data.quat_w_world)
                    rot_9d = rot_mat.reshape(self.num_envs, -1)
                    
                    cam_pose = torch.cat([pos_env, rot_9d], dim=-1)
                    body_cam_pose = self._camera_pose_from_articulation(name, cam_pose)
                    if body_cam_pose is not None:
                        cam_pose = body_cam_pose

                    if name in ["wrist_camera"]:
                        cam_data = cam_sensor.data.output["rgb"]
                        rs_image_list.append(cam_data.unsqueeze(1)) 
                        rs_pose_list.append(cam_pose.unsqueeze(1)) 
                    elif name == "third_view":
                        cam_data = cam_sensor.data.output["rgb"]
                        third_view_image = cam_data
                        third_view_pose = cam_pose
                    else:
                        cam_data = cam_sensor.data.output["rgb"]
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
        if self.object is not None:
            obs["object_pose"] = torch.cat((self.object_pos, self.object_rot), dim=-1)
            obs["object_z_axis"] = quat_to_M6(self.object_rot)
            obs["object_pos"] = self.object_pos
        # -----------------------------------------------------------------------
        # Tag Poses
        # -----------------------------------------------------------------------
        if self.tag_joint_indices.numel() > 0:
            current_center_tag_T = self.hand_dof_pos[:, self.tag_joint_indices]
            obs["current_center_tag_T"] = current_center_tag_T
            if getattr(self.cfg, "enable_fingertip_tag_points", False):
                tag_points = self._compute_fingertip_tag_point_observation(
                    current_center_tag_T
                )
                obs.update(tag_points)
        else:
            obs["current_center_tag_T"] = torch.empty((self.num_envs, 0, 6), device=self.device)
            if getattr(self.cfg, "enable_fingertip_tag_points", False):
                empty = torch.empty((self.num_envs, 0, 4, 3), device=self.device)
                obs["fingertip_tag_points"] = empty
                obs["fingertip_tag_points_current"] = empty
                obs["fingertip_tag_points_local"] = empty
                obs["fingertip_tag_points_canonical"] = empty
                obs["fingertip_tag_points_delta"] = empty
                obs["fingertip_tag_points_env"] = empty
                obs["fingertip_tag_points_current_env"] = empty
                obs["fingertip_tag_points_canonical_env"] = empty
                obs["fingertip_tag_points_delta_env"] = empty
                obs["fingertip_tag_points_world"] = empty
                obs["fingertip_tag_points_current_world"] = empty
                obs["fingertip_tag_points_canonical_world"] = empty
                obs["fingertip_tag_points_delta_world"] = empty

        return obs

    def _compute_fingertip_tag_point_observation(
        self, current_center_tag_T: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return fingertip tag-board corner points in holder-local meters.

        current_center_tag_T is ordered [px, py, pz, rx, ry, rz] per finger.
        The holder joints are small residual tag-board offsets. The canonical
        face is the holder-local Y-Z surface centered
        at cfg.contact_surface_center_link, with outward normal along local -X.
        Env/world variants are also exported so visualization and policy
        experiments can choose either body-attached or pose-independent geometry.
        """
        n_tags = current_center_tag_T.shape[1]
        dtype = current_center_tag_T.dtype
        device = current_center_tag_T.device
        half_w = float(getattr(self.cfg, "fingertip_tag_corner_half_width", 0.01051))
        half_h = float(getattr(self.cfg, "fingertip_tag_corner_half_height", 0.00850))
        center = torch.tensor(
            getattr(self.cfg, "contact_surface_center_link", (-0.0110, -0.002415, 0.0)),
            device=device,
            dtype=dtype,
        )
        canonical = torch.tensor(
            [
                [0.0, -half_w, -half_h],
                [0.0, -half_w, half_h],
                [0.0, half_w, half_h],
                [0.0, half_w, -half_h],
            ],
            device=device,
            dtype=dtype,
        ) + center.view(1, 3)
        canonical = canonical.view(1, 1, 4, 3).expand(self.num_envs, n_tags, -1, -1)

        translation = current_center_tag_T[..., :3]
        rx, ry, rz = current_center_tag_T[..., 3], current_center_tag_T[..., 4], current_center_tag_T[..., 5]
        rotation = self._euler_xyz_to_matrix(rx, ry, rz)
        current_local = torch.matmul(canonical, rotation.transpose(-1, -2)) + translation[:, :, None, :]
        delta_local = current_local - canonical

        obs = {
            # Backward-compatible default remains local holder-frame current points.
            "fingertip_tag_points": current_local,
            "fingertip_tag_points_current": current_local,
            "fingertip_tag_points_local": current_local,
            "fingertip_tag_points_canonical": canonical,
            "fingertip_tag_points_delta": delta_local,
        }

        if self.contact_fingertip_indices.numel() == n_tags:
            # Env/world points should be attached to the actual surface links,
            # whose poses already include the holder residual transforms.
            surface_pos_w = self.hand.data.body_pos_w[:, self.contact_fingertip_indices]
            surface_quat_w = self.hand.data.body_quat_w[:, self.contact_fingertip_indices]
            canonical_world = self._transform_tag_points_to_world(canonical, surface_pos_w, surface_quat_w)
            current_world = canonical_world
            origins = self.scene.env_origins[:, None, None, :]
            canonical_env = canonical_world - origins
            current_env = current_world - origins
            obs.update(
                {
                    "fingertip_tag_points_world": current_world,
                    "fingertip_tag_points_current_world": current_world,
                    "fingertip_tag_points_canonical_world": canonical_world,
                    "fingertip_tag_points_delta_world": current_world - canonical_world,
                    "fingertip_tag_points_env": current_env,
                    "fingertip_tag_points_current_env": current_env,
                    "fingertip_tag_points_canonical_env": canonical_env,
                    "fingertip_tag_points_delta_env": current_env - canonical_env,
                }
            )
        return obs

    @staticmethod
    def _transform_tag_points_to_world(
        points_local: torch.Tensor, anchor_pos_w: torch.Tensor, anchor_quat_w: torch.Tensor
    ) -> torch.Tensor:
        flat_points = points_local.reshape(-1, 3)
        flat_quat = anchor_quat_w[:, :, None, :].expand(-1, -1, points_local.shape[2], -1).reshape(-1, 4)
        flat_pos = anchor_pos_w[:, :, None, :].expand(-1, -1, points_local.shape[2], -1).reshape(-1, 3)
        return (quat_apply(flat_quat, flat_points) + flat_pos).reshape_as(points_local)

    @staticmethod
    def _euler_xyz_to_matrix(rx: torch.Tensor, ry: torch.Tensor, rz: torch.Tensor) -> torch.Tensor:
        cx, sx = torch.cos(rx), torch.sin(rx)
        cy, sy = torch.cos(ry), torch.sin(ry)
        cz, sz = torch.cos(rz), torch.sin(rz)

        zeros = torch.zeros_like(rx)
        ones = torch.ones_like(rx)
        row0_x = torch.stack([ones, zeros, zeros], dim=-1)
        row1_x = torch.stack([zeros, cx, -sx], dim=-1)
        row2_x = torch.stack([zeros, sx, cx], dim=-1)
        rot_x = torch.stack([row0_x, row1_x, row2_x], dim=-2)

        row0_y = torch.stack([cy, zeros, sy], dim=-1)
        row1_y = torch.stack([zeros, ones, zeros], dim=-1)
        row2_y = torch.stack([-sy, zeros, cy], dim=-1)
        rot_y = torch.stack([row0_y, row1_y, row2_y], dim=-2)

        row0_z = torch.stack([cz, -sz, zeros], dim=-1)
        row1_z = torch.stack([sz, cz, zeros], dim=-1)
        row2_z = torch.stack([zeros, zeros, ones], dim=-1)
        rot_z = torch.stack([row0_z, row1_z, row2_z], dim=-2)
        return torch.matmul(rot_z, torch.matmul(rot_y, rot_x))

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        random_env_ids = env_ids[1:]
        # randomizations
        if self.rand_cfg.enable_all or self.rand_cfg.random_lighting:
            self._randomize_env_lighting(
                random_env_ids,
                intensity_range=self.rand_cfg.random_lighting_intensity_range,
                white_jitter=self.rand_cfg.random_lighting_white_jitter,
            )
        if self.rand_cfg.enable_all or self.rand_cfg.random_object_color:
            self._randomize_object_visual_color(random_env_ids)
        if self.rand_cfg.mild_object_color_noise:
            self._randomize_object_visual_color_mild(random_env_ids)
        if self.rand_cfg.enable_all or self.rand_cfg.random_background:
            self._randomize_env_skybox(env_ids) # do not need randomize skybox
            self._randomize_env_floor_color(
                random_env_ids,
                base_gray=self.rand_cfg.random_floor_base_gray,
                delta=self.rand_cfg.random_floor_delta,
            )
        self._reset_object_to_default(env_ids)
        self._reset_hand(env_ids)
        self.successes[env_ids] = 0
        # Initialize obs/buffers immediately
        self._compute_intermediate_values(env_ids)
        self.sim.step()

    def _reset_hand(self, env_ids: torch.Tensor):
        dof_pos = self.init_joint_values.unsqueeze(0).repeat(len(env_ids), 1)
        dof_vel = self.hand.data.default_joint_vel[env_ids] 
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

    def _reset_object_to_default(self, env_ids: torch.Tensor):
        if self.object is None:
            return
        object_default_state = self.object.data.default_root_state.clone()[env_ids]
        object_default_state[:, 0:3] += self.scene.env_origins[env_ids]
        object_default_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_default_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_default_state[:, 7:], env_ids)


    def _compute_intermediate_values(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
            
        self.hand_dof_pos = self.hand.data.joint_pos
        self.actuated_dof_pos = self.hand_dof_pos[:, self.actuated_dof_indices]
        self.hand_dof_vel = self.hand.data.joint_vel

        if self.object is not None:
            self.object_pos = self.object.data.root_pos_w - self.scene.env_origins
            self.object_rot = self.object.data.root_quat_w
            self.object_velocities = self.object.data.root_vel_w
            self.object_linvel = self.object.data.root_lin_vel_w
            self.object_angvel = self.object.data.root_ang_vel_w

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
        if self.cfg.replay_mode:
            # Replay writes recorded joint/object poses directly; avoid letting
            # physics drift those poses between recorded frames. Update
            # articulation kinematics and Fabric before rendering, otherwise RTX
            # cameras can see the previous link transforms while tensors already
            # contain the requested replay state.
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.forward()
            self.sim.render()
            self.scene.update(dt=0.0)
        else:
            # Match Isaac Lab's DirectRLEnv stepping: targets only become visible
            # to cameras after the simulator advances. Rendering without sim.step
            # leaves policy eval videos effectively stuck at the reset frame.
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                self._apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render()
                self.scene.update(dt=self.physics_dt)

        # post-step:
        # -- update env counters (used for curriculum generation)
        self.episode_length_buf += 1  # step in current episode (per env)
        self.common_step_counter += 1  # total step (common for all envs)

        if self.cfg.replay_mode:
            self.reset_terminated[:] = False
            self.reset_time_outs[:] = False
            self.reset_buf[:] = False
            if not hasattr(self, "reward_buf"):
                self.reward_buf = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            self.reward_buf[:] = 0.0
        else:
            self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
            self.reset_buf = self.reset_terminated | self.reset_time_outs
            self.reward_buf = self._get_rewards()

            # -- reset envs that terminated/timed-out and log the episode information
            reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                self._reset_idx(reset_env_ids)
                # if sensors are added to the scene, make sure we render to reflect changes in reset
                num_rerenders_on_reset = int(getattr(self.cfg, "num_rerenders_on_reset", 0))
                if self.sim.has_rtx_sensors() and num_rerenders_on_reset > 0:
                    for _ in range(num_rerenders_on_reset):
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

    def _object_prim_name(self) -> str | None:
        object_cfg = getattr(self.cfg, "object_cfg", None)
        if object_cfg is None:
            return None
        return str(object_cfg.prim_path).rstrip("/").split("/")[-1]

    def _randomize_object_visual_color(self, env_ids=None):
        if self.object is None:
            return
        stage = omni.usd.get_context().get_stage()

        if env_ids is None:
            env_ids = range(self.num_envs)
        object_prim_name = self._object_prim_name()
        if object_prim_name is None:
            return

        for i in env_ids:
            color = Gf.Vec3f(
                float(torch.rand((), device=self.device).item()),
                float(torch.rand((), device=self.device).item()),
                float(torch.rand((), device=self.device).item()),
            )
            self._set_object_visual_color(stage, int(i), object_prim_name, color)

    def _randomize_object_visual_color_mild(self, env_ids=None):
        if self.object is None:
            return
        stage = omni.usd.get_context().get_stage()

        if env_ids is None:
            env_ids = range(self.num_envs)
        object_prim_name = self._object_prim_name()
        if object_prim_name is None:
            return

        base = torch.tensor(self.rand_cfg.mild_object_color_base, device=self.device, dtype=torch.float32)
        scale = float(self.rand_cfg.mild_object_color_noise_scale)

        for i in env_ids:
            noise = torch.empty(3, device=self.device).uniform_(-scale, scale)
            color_tensor = torch.clamp(base + noise, 0.0, 1.0)
            color = Gf.Vec3f(*[float(value) for value in color_tensor.tolist()])
            self._set_object_visual_color(stage, int(i), object_prim_name, color)

    def _set_object_visual_color(self, stage, env_id: int, object_prim_name: str, color: Gf.Vec3f) -> None:
        object_root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/{object_prim_name}")
        if not object_root.IsValid():
            return
        for prim in Usd.PrimRange(object_root):
            if prim.IsA(UsdGeom.Gprim):
                gprim = UsdGeom.Gprim(prim)
                color_attr = gprim.GetDisplayColorAttr()
                if not color_attr:
                    color_attr = gprim.CreateDisplayColorAttr()
                color_attr.Set([color])
            if prim.GetTypeName() == "Shader":
                attr = prim.GetAttribute("inputs:diffuseColor")
                if attr:
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
