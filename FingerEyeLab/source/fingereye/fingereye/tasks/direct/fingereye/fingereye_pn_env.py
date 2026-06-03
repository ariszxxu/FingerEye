from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_physics_material, get_current_stage
from isaaclab.utils.math import compute_pose_error, quat_apply, sample_uniform
from pxr import Gf, Usd, UsdGeom

from .env_tools import ensure_floor_uv, quat_to_M6
from .fingereye_env import FingerEyeLabEnv
from .fingereye_pn_env_cfg import FingerEyePNLabEnvCfg


def transform_points(points: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    points_was_batched = points.ndim == 3
    T_was_batched = T.ndim == 3
    if points.ndim == 2:
        points = points.unsqueeze(0)
    if T.ndim == 2:
        T = T.unsqueeze(0)
    if points.shape[0] == 1 and T.shape[0] > 1:
        points = points.expand(T.shape[0], -1, -1)
    if T.shape[0] == 1 and points.shape[0] > 1:
        T = T.expand(points.shape[0], -1, -1)
    transformed = points @ T[:, :3, :3].transpose(1, 2) + T[:, None, :3, 3]
    if not points_was_batched and not T_was_batched:
        return transformed.squeeze(0)
    return transformed


def make_transform(pos: torch.Tensor, quat_wxyz: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = quat_wxyz.unbind(dim=-1)
    two = 2.0
    rot = torch.empty(pos.shape[:-1] + (3, 3), device=pos.device, dtype=pos.dtype)
    rot[..., 0, 0] = 1 - two * (qy * qy + qz * qz)
    rot[..., 0, 1] = two * (qx * qy - qz * qw)
    rot[..., 0, 2] = two * (qx * qz + qy * qw)
    rot[..., 1, 0] = two * (qx * qy + qz * qw)
    rot[..., 1, 1] = 1 - two * (qx * qx + qz * qz)
    rot[..., 1, 2] = two * (qy * qz - qx * qw)
    rot[..., 2, 0] = two * (qx * qz - qy * qw)
    rot[..., 2, 1] = two * (qy * qz + qx * qw)
    rot[..., 2, 2] = 1 - two * (qx * qx + qy * qy)
    transform = torch.eye(4, device=pos.device, dtype=pos.dtype).expand(pos.shape[0], 4, 4).clone()
    transform[:, :3, :3] = rot
    transform[:, :3, 3] = pos
    return transform


def compute_cylinder_surface_distance(points_local: torch.Tensor, radius: float, thickness: float) -> torch.Tensor:
    radial = torch.linalg.norm(points_local[..., :2], dim=-1)
    axial = torch.abs(points_local[..., 2])
    q_radial = radial - radius
    q_axial = axial - 0.5 * thickness
    outside = torch.linalg.norm(torch.clamp(torch.stack((q_radial, q_axial), dim=-1), min=0.0), dim=-1)
    inside = torch.minimum(torch.maximum(q_radial, q_axial), torch.zeros_like(q_radial))
    return torch.abs(outside + inside)


class FingerEyePNLabEnv(FingerEyeLabEnv):
    """Final pick-nut task with no screw/bolt asset."""

    cfg: FingerEyePNLabEnvCfg

    def __init__(self, cfg: FingerEyePNLabEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.hp_arm_joint_indices = torch.tensor(
            [self.hand.joint_names.index(f"joint{i}") for i in range(1, 8)],
            dtype=torch.long,
            device=self.device,
        )
        self.hp_ee_body_idx = self.hand.body_names.index(self.cfg.hp_ee_body_name)
        self.hp_ee_jacobi_idx = self.hp_ee_body_idx - 1 if self.hand.is_fixed_base else self.hp_ee_body_idx
        self.hp_ee_reset_pos = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.hp_ee_reset_quat = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.hp_ee_reset_quat[:, 0] = 1.0
        self.hp_ee_reset_pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.pn_tip_body_names = list(getattr(self.cfg, "contact_fingertip_link_names", ["fingertip", "thumb_soft_ring"]))
        if self.contact_fingertip_indices.numel() != 2:
            raise RuntimeError(
                "PN expects two fingertip surface links. "
                f"Configured links={self.pn_tip_body_names}, resolved={self.contact_fingertip_indices.tolist()}."
            )
        self.pn_tip_body_indices = self.contact_fingertip_indices
        self.pn_tip_jacobi_indices = self.pn_tip_body_indices - 1 if self.hand.is_fixed_base else self.pn_tip_body_indices
        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._reset_pn_expert_state(env_ids)
        self._reset_sort_expert_state(env_ids)

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.setup_gray_ground()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["nut"] = self.object

        stage = get_current_stage()
        for env_id in range(self.num_envs):
            floor_prim_path = f"/World/envs/env_{env_id}/Background/floor"
            ensure_floor_uv(stage, floor_prim_path)
            self._setup_pn_pick_markers(stage, env_id)
            self._bind_pn_pick_grasp_physics_materials(stage, env_id)

        if self.rand_cfg.enable_all or self.rand_cfg.random_lighting:
            self._setup_env_lights()
        else:
            light_cfg = sim_utils.DomeLightCfg(
                intensity=2000.0,
                color=(0.75, 0.75, 0.75),
                visible_in_primary_ray=self.rand_cfg.visible_in_primary_ray,
            )
            light_cfg.func("/World/Light", light_cfg)
        if self.rand_cfg.enable_all or self.rand_cfg.random_background:
            self._setup_env_backgrounds()
            for env_id in range(self.num_envs):
                self._setup_env_skybox_and_light_for_env(env_id)

        if self.cfg.enable_cameras:
            self._camera_configs = {
                "wrist_camera": self.cfg.cam_wrist if hasattr(self.cfg, "cam_wrist") else None,
                "index_tip": self.cfg.cam_index_tip if hasattr(self.cfg, "cam_index_tip") else None,
                "index_root": self.cfg.cam_index_root if hasattr(self.cfg, "cam_index_root") else None,
                "thumb_tip": self.cfg.cam_thumb_tip if hasattr(self.cfg, "cam_thumb_tip") else None,
                "thumb_root": self.cfg.cam_thumb_root if hasattr(self.cfg, "cam_thumb_root") else None,
                "third_view": self.cfg.cam_third_view if hasattr(self.cfg, "cam_third_view") else None,
            }
            self._active_cameras = {}
            self._camera_body_names = {}
            self._camera_body_indices = {}
            self._camera_local_pos = {}
            self._camera_local_rot = {}
            self._camera_runtime_local_pos = {}
            self._camera_runtime_local_rot = {}
            for name in self.cfg.camera_name_list:
                if name in self._camera_configs:
                    camera_cfg = self._camera_configs[name]
                    print(f"[INFO] Spawning Camera: {name} ({camera_cfg.width}x{camera_cfg.height})")
                    if camera_cfg.spawn is None:
                        self._clone_camera_with_standard_xform_ops(stage, camera_cfg)
                    sensor = TiledCamera(camera_cfg)
                    self.scene.sensors[name] = sensor
                    self._active_cameras[name] = sensor
                    self._register_camera_body_pose_source(name, camera_cfg, stage)

    def _bind_pn_pick_grasp_physics_materials(self, stage, env_id: int):
        nut_material_path = f"/World/envs/env_{env_id}/PNPickNutGripMaterial"
        nut_material_cfg = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(getattr(self.cfg, "pn_pick_nut_static_friction", 1.0)),
            dynamic_friction=float(getattr(self.cfg, "pn_pick_nut_dynamic_friction", 1.0)),
            friction_combine_mode="max",
        )
        nut_material_cfg.func(nut_material_path, nut_material_cfg)
        nut_path = f"/World/envs/env_{env_id}/Nut"
        if stage.GetPrimAtPath(nut_path).IsValid():
            bind_physics_material(nut_path, nut_material_path, stage=stage)

        if bool(getattr(self.cfg, "pn_pick_bind_robot_physics_material", False)):
            robot_material_path = f"/World/envs/env_{env_id}/PNPickRobotGripMaterial"
            robot_material_cfg = sim_utils.RigidBodyMaterialCfg(
                static_friction=float(getattr(self.cfg, "pn_pick_robot_static_friction", 1.0)),
                dynamic_friction=float(getattr(self.cfg, "pn_pick_robot_dynamic_friction", 1.0)),
                friction_combine_mode="max",
            )
            robot_material_cfg.func(robot_material_path, robot_material_cfg)
            robot_path = f"/World/envs/env_{env_id}/Robot"
            robot_prim = stage.GetPrimAtPath(robot_path)
            if robot_prim.IsValid():
                for prim in Usd.PrimRange(robot_prim):
                    if prim.IsInstance():
                        prim.SetInstanceable(False)
                bind_physics_material(robot_path, robot_material_path, stage=stage)

    def _setup_pn_pick_markers(self, stage, env_id: int):
        env_path = f"/World/envs/env_{env_id}"
        z = float(self.cfg.table_z) + 0.00015
        if bool(getattr(self.cfg, "show_pick_range_marker", True)):
            self._make_range_box(
                stage,
                f"{env_path}/Debug/pn_pick_pick_range",
                float(self.cfg.nut_x_min),
                float(self.cfg.nut_x_max),
                float(self.cfg.nut_y_min),
                float(self.cfg.nut_y_max),
                z,
                Gf.Vec3f(1.0, 0.0, 0.0),
            )
        if bool(getattr(self.cfg, "show_pn_pick_target_markers", True)):
            r = 0.007
            self._make_target_marker(
                stage,
                f"{env_path}/Debug/pn_pick_light_target",
                self.cfg.pn_pick_light_target_xy,
                z,
                r,
                Gf.Vec3f(0.05, 0.35, 1.0),
            )
            self._make_target_marker(
                stage,
                f"{env_path}/Debug/pn_pick_heavy_target",
                self.cfg.pn_pick_heavy_target_xy,
                z,
                r,
                Gf.Vec3f(1.0, 0.65, 0.05),
            )

    def _make_range_box(self, stage, prim_path: str, x_min: float, x_max: float, y_min: float, y_max: float, z: float, color):
        thickness = 0.002
        corners = (
            (0.5 * (x_min + x_max), y_min, z, x_max - x_min, thickness, thickness),
            (0.5 * (x_min + x_max), y_max, z, x_max - x_min, thickness, thickness),
            (x_min, 0.5 * (y_min + y_max), z, thickness, y_max - y_min, thickness),
            (x_max, 0.5 * (y_min + y_max), z, thickness, y_max - y_min, thickness),
        )
        for i, (x, y, z_i, sx, sy, sz) in enumerate(corners):
            cube = UsdGeom.Cube.Define(stage, f"{prim_path}_{i}")
            cube.CreateSizeAttr(1.0)
            prim = cube.GetPrim()
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z_i))
            xform.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr().Set([color])
            gprim.CreateDisplayOpacityAttr().Set([1.0])

    def _make_target_marker(self, stage, prim_path: str, xy, z: float, radius: float, color):
        cylinder = UsdGeom.Cylinder.Define(stage, prim_path)
        cylinder.CreateRadiusAttr(radius)
        cylinder.CreateHeightAttr(0.001)
        prim = cylinder.GetPrim()
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(float(xy[0]), float(xy[1]), z))
        gprim = UsdGeom.Gprim(prim)
        gprim.CreateDisplayColorAttr().Set([color])
        gprim.CreateDisplayOpacityAttr().Set([0.9])

    def _compute_pn_arm_joint_targets_absolute(self, eef_target_env: torch.Tensor) -> torch.Tensor:
        ee_pos_w = self.hand.data.body_pos_w[:, self.hp_ee_body_idx]
        ee_quat_w = self.hand.data.body_quat_w[:, self.hp_ee_body_idx]
        ee_target_w = eef_target_env + self.scene.env_origins
        position_error, axis_angle_error = compute_pose_error(
            ee_pos_w,
            ee_quat_w,
            ee_target_w,
            self.hp_ee_reset_quat,
            rot_error_type="axis_angle",
        )
        pos_axis_gains = torch.tensor(self.cfg.hp_ik_pos_axis_gains, device=self.device, dtype=position_error.dtype)
        pose_error = torch.cat((position_error * pos_axis_gains, axis_angle_error * float(self.cfg.hp_ik_rot_gain)), dim=-1)

        jacobians = self.hand.root_physx_view.get_jacobians()
        jacobian = jacobians[:, self.hp_ee_jacobi_idx, 0:6, :][:, :, self.hp_arm_joint_indices]
        jacobian_t = jacobian.transpose(1, 2)
        lambda_sq = float(self.cfg.hp_ik_damping) ** 2
        eye = torch.eye(6, device=self.device, dtype=jacobian.dtype).unsqueeze(0).repeat(self.num_envs, 1, 1)
        delta_q = jacobian_t @ torch.linalg.solve(jacobian @ jacobian_t + lambda_sq * eye, pose_error.unsqueeze(-1))
        delta_q = float(self.cfg.hp_ik_gain) * delta_q.squeeze(-1)
        delta_q = torch.clamp(delta_q, -float(self.cfg.hp_ik_max_joint_delta), float(self.cfg.hp_ik_max_joint_delta))
        joint_pos = self.hand.data.joint_pos[:, self.hp_arm_joint_indices]
        target = joint_pos + delta_q
        lower = self.hand_dof_lower_limits[:, self.hp_arm_joint_indices]
        upper = self.hand_dof_upper_limits[:, self.hp_arm_joint_indices]
        return torch.clamp(target, lower, upper)

    def _reset_nut_asset(self, env_ids: torch.Tensor):
        n = len(env_ids)
        nut_x = sample_uniform(self.cfg.nut_x_min, self.cfg.nut_x_max, (n,), device=self.device)
        nut_y = sample_uniform(self.cfg.nut_y_min, self.cfg.nut_y_max, (n,), device=self.device)
        nut_state = self.object.data.default_root_state.clone()[env_ids]
        nut_state[:, 0] = nut_x
        nut_state[:, 1] = nut_y
        nut_state[:, 2] = float(getattr(self.cfg, "nut_reset_center_z", self.cfg.nut_height * 0.5))
        yaw = torch.full((n,), float(self.cfg.nut_init_yaw_deg), dtype=torch.float32, device=self.device) * torch.pi / 180.0
        nut_state[:, 3] = torch.cos(0.5 * yaw)
        nut_state[:, 4] = 0.0
        nut_state[:, 5] = 0.0
        nut_state[:, 6] = torch.sin(0.5 * yaw)
        nut_state[:, 0:3] += self.scene.env_origins[env_ids]
        nut_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(nut_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(nut_state[:, 7:], env_ids)

    def _enforce_nut_reset_yaw(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return
        nut_state = self.object.data.root_state_w.clone()[env_ids]
        yaw = torch.full((len(env_ids),), float(self.cfg.nut_init_yaw_deg), dtype=torch.float32, device=self.device)
        yaw = yaw * torch.pi / 180.0
        nut_state[:, 2] = self.scene.env_origins[env_ids, 2] + float(getattr(self.cfg, "nut_reset_center_z", 0.0))
        nut_state[:, 3] = torch.cos(0.5 * yaw)
        nut_state[:, 4] = 0.0
        nut_state[:, 5] = 0.0
        nut_state[:, 6] = torch.sin(0.5 * yaw)
        nut_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(nut_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(nut_state[:, 7:], env_ids)

    def _reset_hand(self, env_ids: torch.Tensor):
        dof_pos = self.init_joint_values.unsqueeze(0).repeat(len(env_ids), 1)
        dof_vel = self.hand.data.default_joint_vel[env_ids]
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)
        self.hp_ee_reset_pos[env_ids] = self.hand.data.body_pos_w[env_ids, self.hp_ee_body_idx] - self.scene.env_origins[env_ids]
        self.hp_ee_reset_quat[env_ids] = self.hand.data.body_quat_w[env_ids, self.hp_ee_body_idx]
        self.hp_ee_reset_pending[env_ids] = True

    def _compute_pn_success(self, height: float | None = None) -> torch.Tensor:
        height_threshold = float(self.cfg.pn_success_height if height is None else height) - float(
            getattr(self.cfg, "pn_success_height_tolerance", 0.0)
        )
        nut_top_z = self.object_pos[:, 2] + 0.5 * float(self.cfg.nut_height)
        nut_lifted = nut_top_z >= height_threshold
        tip_pos = self.hand.data.body_pos_w[:, self.pn_tip_body_indices] - self.scene.env_origins[:, None, :]
        tips_lifted = tip_pos[:, :, 2].min(dim=1).values > (
            float(self.cfg.table_z)
            + 0.5 * float(self.cfg.nut_height)
            + float(getattr(self.cfg, "pn_success_tip_height_margin", 0.001))
        )
        return nut_lifted & tips_lifted

    def _compute_sort_between_fingertips(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tip_pos = self.hand.data.body_pos_w[:, self.pn_tip_body_indices] - self.scene.env_origins[:, None, :]
        nut_xy = self.object_pos[:, :2]
        tip_a = tip_pos[:, 0, :2]
        tip_b = tip_pos[:, 1, :2]
        tip_axis = tip_b - tip_a
        tip_axis_sq = tip_axis.square().sum(dim=-1).clamp_min(1e-8)
        along = ((nut_xy - tip_a) * tip_axis).sum(dim=-1) / tip_axis_sq
        closest_xy = tip_a + along.clamp(0.0, 1.0).unsqueeze(-1) * tip_axis
        xy_error = torch.linalg.vector_norm(nut_xy - closest_xy, dim=-1)
        between_tips = (along > 0.05) & (along < 0.95)
        near_grasp_line = xy_error < (
            0.5 * float(self.cfg.nut_diameter) + float(getattr(self.cfg, "pn_success_xy_margin", 0.004))
        )
        return between_tips, near_grasp_line, xy_error

    def _compute_sort_grasped(self) -> torch.Tensor:
        between_tips, near_grasp_line, _ = self._compute_sort_between_fingertips()
        tip_pos = self.hand.data.body_pos_w[:, self.pn_tip_body_indices] - self.scene.env_origins[:, None, :]
        tips_lifted = tip_pos[:, :, 2].min(dim=1).values > (
            float(self.cfg.table_z)
            + 0.5 * float(self.cfg.nut_height)
            + float(getattr(self.cfg, "pn_success_tip_height_margin", 0.001))
        )
        return between_tips & near_grasp_line & tips_lifted

    def _compute_sort_pick_success(self) -> torch.Tensor:
        return self._compute_pn_success(float(self.cfg.pn_expert_success_height)) & self._compute_sort_grasped()

    def _get_sort_target_xy(self) -> torch.Tensor:
        light_xy = torch.tensor(self.cfg.pn_pick_light_target_xy, dtype=self.object_pos.dtype, device=self.device)
        heavy_xy = torch.tensor(self.cfg.pn_pick_heavy_target_xy, dtype=self.object_pos.dtype, device=self.device)
        heavy = torch.full(
            (self.num_envs, 1),
            float(getattr(self.cfg, "pn_bd_nut_mass", 0.0)) >= float(getattr(self.cfg, "pn_pick_heavy_mass_threshold", 0.030)),
            dtype=torch.bool,
            device=self.device,
        )
        return torch.where(heavy, heavy_xy.unsqueeze(0), light_xy.unsqueeze(0))

    def _pn_pick_phase_ids(self) -> dict[str, int]:
        sensing_steps = int(getattr(self.cfg, "pn_pick_sensing_hold_steps", 0))
        if sensing_steps > 0:
            return {"sensing": 1, "move": 2, "lower": 3, "release": 4, "retreat": 5, "done": 6}
        return {"sensing": -1, "move": 1, "lower": 2, "release": 3, "retreat": 4, "done": 5}

    def _compute_task_success(self) -> torch.Tensor:
        return self._compute_pn_success(float(self.cfg.pn_success_height)) & self._compute_sort_grasped()

    def compute_expert_record_success(self) -> torch.Tensor:
        return self._compute_sort_pick_success()

    def _reset_pn_expert_state(self, env_ids: torch.Tensor, resample_noise: bool = True):
        n_envs = self.num_envs
        if not hasattr(self, "pn_expert_phase"):
            self.pn_expert_phase = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.pn_expert_eef_target = torch.zeros((n_envs, 3), dtype=torch.float32, device=self.device)
            self.pn_expert_initial_eef_target = torch.zeros((n_envs, 3), dtype=torch.float32, device=self.device)
            self.pn_expert_eef_base_z = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
            self.pn_expert_leap_target = torch.zeros((n_envs, len(self.control_dof_indices)), dtype=torch.float32, device=self.device)
            self.pn_expert_noise = torch.zeros((n_envs, 3), dtype=torch.float32, device=self.device)
            self.pn_expert_attempt_noise = torch.zeros((n_envs, 3, 3), dtype=torch.float32, device=self.device)
            self.pn_expert_attempt_count = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.pn_expert_overclose_count = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.pn_expert_phase_step_count = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.pn_expert_lift_target_z = torch.zeros(n_envs, dtype=torch.float32, device=self.device)
            self.pn_expert_lift_xy = torch.zeros((n_envs, 2), dtype=torch.float32, device=self.device)
            self.pn_expert_lift_xy_valid = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
            self.pn_expert_grasp_center = torch.zeros((n_envs, 3), dtype=torch.float32, device=self.device)
            self.pn_expert_initial_leap_target = torch.zeros((n_envs, len(self.control_dof_indices)), dtype=torch.float32, device=self.device)
            self.pn_expert_xy_stage_offsets = torch.zeros((n_envs, 3, 2), dtype=torch.float32, device=self.device)
            self.pn_expert_timing_offset = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.pn_bd_init_wrist_offset = torch.zeros((n_envs, 3), dtype=torch.float32, device=self.device)
            self.pn_bd_goto_nut_curve_offset = torch.zeros((n_envs, 3), dtype=torch.float32, device=self.device)
        self.pn_expert_phase[env_ids] = 0
        self.pn_expert_attempt_count[env_ids] = 0
        self.pn_expert_overclose_count[env_ids] = 0
        self.pn_expert_phase_step_count[env_ids] = 0
        self.pn_expert_lift_target_z[env_ids] = 0.0
        self.pn_expert_lift_xy[env_ids] = 0.0
        self.pn_expert_lift_xy_valid[env_ids] = False
        self.pn_expert_grasp_center[env_ids] = 0.0
        self.pn_expert_timing_offset[env_ids] = 0
        if hasattr(self, "hand") and hasattr(self.hand, "data"):
            self.pn_expert_eef_target[env_ids] = self.hand.data.body_pos_w[env_ids, self.hp_ee_body_idx] - self.scene.env_origins[env_ids]
            self.pn_expert_initial_eef_target[env_ids] = self.pn_expert_eef_target[env_ids]
            self.pn_expert_eef_base_z[env_ids] = self.pn_expert_eef_target[env_ids, 2]
            self.pn_expert_leap_target[env_ids] = self.hand.data.joint_pos[env_ids][:, self.control_dof_indices]
            self.pn_expert_initial_leap_target[env_ids] = self.pn_expert_leap_target[env_ids]
        if resample_noise:
            self._resample_pn_expert_noise(env_ids)
        if resample_noise and bool(getattr(self.cfg, "pn_bd_expert_diversity_enabled", False)):
            self.pn_expert_initial_eef_target[env_ids] = (
                self.pn_expert_initial_eef_target[env_ids] + self.pn_bd_init_wrist_offset[env_ids]
            )

    def _reset_sort_expert_state(self, env_ids: torch.Tensor):
        n_envs = self.num_envs
        if not hasattr(self, "sort_phase"):
            self.sort_phase = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.sort_phase_step_count = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.sort_phase_arrived_count = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.sort_move_eef_target = torch.zeros((n_envs, 3), dtype=torch.float32, device=self.device)
            self.sort_lower_eef_target = torch.zeros((n_envs, 3), dtype=torch.float32, device=self.device)
            self.sort_retreat_eef_target = torch.zeros((n_envs, 3), dtype=torch.float32, device=self.device)
            self.sort_release_start_q = torch.zeros((n_envs, len(self.control_dof_indices)), dtype=torch.float32, device=self.device)
            self.sort_return_start_q = torch.zeros((n_envs, len(self.control_dof_indices)), dtype=torch.float32, device=self.device)
            self.sort_hold_eef_rel_xy = torch.zeros((n_envs, 2), dtype=torch.float32, device=self.device)
            self.sort_picked = torch.zeros(n_envs, dtype=torch.bool, device=self.device)
            self.sort_reset_counter = torch.zeros(n_envs, dtype=torch.long, device=self.device)
            self.sort_expert_event = torch.zeros(n_envs, dtype=torch.long, device=self.device)
        self.sort_phase[env_ids] = 0
        self.sort_phase_step_count[env_ids] = 0
        self.sort_phase_arrived_count[env_ids] = 0
        self.sort_move_eef_target[env_ids] = 0.0
        self.sort_lower_eef_target[env_ids] = 0.0
        self.sort_retreat_eef_target[env_ids] = 0.0
        self.sort_release_start_q[env_ids] = 0.0
        self.sort_return_start_q[env_ids] = 0.0
        self.sort_hold_eef_rel_xy[env_ids] = 0.0
        self.sort_picked[env_ids] = False
        self.sort_expert_event[env_ids] = 0

    def _get_pn_expert_contact_points_env(self) -> torch.Tensor:
        local = self._get_pn_expert_contact_points_link()
        body_pos = self.hand.data.body_pos_w[:, self.pn_tip_body_indices] - self.scene.env_origins[:, None, :]
        body_quat = self.hand.data.body_quat_w[:, self.pn_tip_body_indices]
        return body_pos + quat_apply(body_quat.reshape(-1, 4), local.reshape(-1, 3)).reshape_as(body_pos)

    def _get_pn_expert_contact_points_link(self) -> torch.Tensor:
        link_pos_env = self.hand.data.body_pos_w[:, self.pn_tip_body_indices] - self.scene.env_origins[:, None, :]
        link_quat = self.hand.data.body_quat_w[:, self.pn_tip_body_indices]
        T_env_from_coin = make_transform(self.object_pos, self.object_rot)
        T_coin_from_env = torch.linalg.inv(T_env_from_coin)
        grid_link = self._pn_contact_grid_link(dtype=link_pos_env.dtype)
        selected = []
        for finger_id in range(2):
            T_env_from_link = make_transform(link_pos_env[:, finger_id], link_quat[:, finger_id])
            grid_env = transform_points(grid_link, T_env_from_link)
            grid_coin = transform_points(grid_env, T_coin_from_env)
            dist = compute_cylinder_surface_distance(
                grid_coin,
                radius=self.cfg.contact_coin_radius,
                thickness=self.cfg.contact_coin_thickness,
            )
            selected.append(grid_link[dist.argmin(dim=-1)])
        return torch.stack(selected, dim=1)

    def _pn_contact_grid_link(self, dtype: torch.dtype) -> torch.Tensor:
        center = torch.tensor(self.cfg.contact_surface_center_link, dtype=dtype, device=self.device)
        ys = torch.linspace(
            center[1] - 0.5 * float(self.cfg.contact_surface_width),
            center[1] + 0.5 * float(self.cfg.contact_surface_width),
            int(self.cfg.contact_grid_width),
            dtype=dtype,
            device=self.device,
        )
        zs = torch.linspace(
            center[2] - 0.5 * float(self.cfg.contact_surface_height),
            center[2] + 0.5 * float(self.cfg.contact_surface_height),
            int(self.cfg.contact_grid_height),
            dtype=dtype,
            device=self.device,
        )
        zz, yy = torch.meshgrid(zs, ys, indexing="ij")
        xx = torch.full_like(yy, center[0])
        return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)

    def _get_pn_expert_surface_distances(self) -> torch.Tensor:
        link_pos_env = self.hand.data.body_pos_w[:, self.pn_tip_body_indices] - self.scene.env_origins[:, None, :]
        link_quat = self.hand.data.body_quat_w[:, self.pn_tip_body_indices]
        T_coin_from_env = torch.linalg.inv(make_transform(self.object_pos, self.object_rot))
        local = self._get_pn_expert_contact_points_link()
        distances = []
        for finger_id in range(2):
            T_env_from_link = make_transform(link_pos_env[:, finger_id], link_quat[:, finger_id])
            point_env = transform_points(local[:, finger_id], T_env_from_link)
            if point_env.ndim == 3:
                point_env = point_env[:, 0, :]
            point_coin = transform_points(point_env, T_coin_from_env)
            if point_coin.ndim == 3:
                point_coin = point_coin[:, 0, :]
            distances.append(
                compute_cylinder_surface_distance(
                    point_coin[:, None, :],
                    radius=self.cfg.contact_coin_radius,
                    thickness=self.cfg.contact_coin_thickness,
                )[:, 0]
            )
        return torch.stack(distances, dim=1)

    @staticmethod
    def _skew(v: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(v[..., 0])
        return torch.stack(
            (
                torch.stack((zeros, -v[..., 2], v[..., 1]), dim=-1),
                torch.stack((v[..., 2], zeros, -v[..., 0]), dim=-1),
                torch.stack((-v[..., 1], v[..., 0], zeros), dim=-1),
            ),
            dim=-2,
        )

    def _resample_pn_expert_noise(self, env_ids: torch.Tensor):
        std = torch.tensor(self.cfg.pn_expert_noise_std, device=self.device, dtype=torch.float32)
        clip = float(self.cfg.pn_expert_noise_clip)
        noise = torch.randn((len(env_ids), 3, 3), dtype=torch.float32, device=self.device) * std.view(1, 1, 3)
        self.pn_expert_attempt_noise[env_ids] = torch.clamp(noise, -clip * std.view(1, 1, 3), clip * std.view(1, 1, 3))
        self.pn_expert_noise[env_ids] = self.pn_expert_attempt_noise[env_ids, 0]
        ranges = torch.tensor(self.cfg.pn_expert_xy_stage_offset_ranges, dtype=torch.float32, device=self.device).view(1, 3, 1)
        rough = torch.empty((len(env_ids), 3, 2), dtype=torch.float32, device=self.device)
        rough.uniform_(-1.0, 1.0)
        self.pn_expert_xy_stage_offsets[env_ids] = rough * ranges
        self._resample_pn_bd_pick_diversity(env_ids)

    def _sample_symmetric_vec(self, n: int, ranges) -> torch.Tensor:
        ranges_t = torch.tensor(ranges, dtype=torch.float32, device=self.device).view(1, -1)
        values = torch.empty((n, ranges_t.shape[-1]), dtype=torch.float32, device=self.device)
        values.uniform_(-1.0, 1.0)
        return values * ranges_t

    def _resample_pn_bd_pick_diversity(self, env_ids: torch.Tensor):
        if not hasattr(self, "pn_bd_init_wrist_offset"):
            return
        self.pn_bd_init_wrist_offset[env_ids] = 0.0
        self.pn_bd_goto_nut_curve_offset[env_ids] = 0.0
        if not bool(getattr(self.cfg, "pn_bd_expert_diversity_enabled", False)):
            return

        n = len(env_ids)
        if bool(getattr(self.cfg, "pn_bd_init_wrist_xy_from_pick_box", False)):
            x_center = 0.5 * (float(self.cfg.nut_x_min) + float(self.cfg.nut_x_max))
            y_center = 0.5 * (float(self.cfg.nut_y_min) + float(self.cfg.nut_y_max))
            xy_min = torch.tensor(
                [float(self.cfg.nut_x_min) - x_center, float(self.cfg.nut_y_min) - y_center],
                dtype=torch.float32,
                device=self.device,
            )
            xy_max = torch.tensor(
                [float(self.cfg.nut_x_max) - x_center, float(self.cfg.nut_y_max) - y_center],
                dtype=torch.float32,
                device=self.device,
            )
            xy = torch.rand((n, 2), dtype=torch.float32, device=self.device) * (xy_max - xy_min) + xy_min
            self.pn_bd_init_wrist_offset[env_ids, :2] = xy

        z_offset_range = getattr(self.cfg, "pn_bd_init_wrist_z_offset_range", None)
        if z_offset_range is not None:
            z_min, z_max = [float(v) for v in z_offset_range]
            if z_max > z_min:
                z = torch.empty((n,), dtype=torch.float32, device=self.device)
                z.uniform_(z_min, z_max)
                self.pn_bd_init_wrist_offset[env_ids, 2] = z
        else:
            z_range = float(getattr(self.cfg, "pn_bd_init_wrist_z_noise_range", 0.0))
            if z_range > 0.0:
                z = torch.empty((n,), dtype=torch.float32, device=self.device)
                z.uniform_(-z_range, z_range)
                self.pn_bd_init_wrist_offset[env_ids, 2] = z

        self.pn_bd_goto_nut_curve_offset[env_ids] = self._sample_symmetric_vec(
            n, getattr(self.cfg, "pn_bd_goto_nut_curve_range", (0.0, 0.0, 0.0))
        )

    def _apply_pn_bd_diverse_initial_wrist_pose(self, env_ids: torch.Tensor):
        if not bool(getattr(self.cfg, "pn_bd_expert_diversity_enabled", False)):
            return
        if env_ids.numel() == 0:
            return
        target = self.pn_expert_initial_eef_target.clone()
        old_reset_pos = self.hp_ee_reset_pos.clone()
        old_reset_quat = self.hp_ee_reset_quat.clone()
        self.hp_ee_reset_pos[:] = self.hand.data.body_pos_w[:, self.hp_ee_body_idx] - self.scene.env_origins
        self.hp_ee_reset_quat[:] = self.hand.data.body_quat_w[:, self.hp_ee_body_idx]
        for _ in range(int(getattr(self.cfg, "pn_bd_init_wrist_ik_iters", 8))):
            arm_targets = self._compute_pn_arm_joint_targets_absolute(target)
            dof_pos = self.hand.data.joint_pos.clone()
            dof_vel = torch.zeros_like(dof_pos)
            dof_pos[env_ids.unsqueeze(-1), self.hp_arm_joint_indices.unsqueeze(0)] = arm_targets[env_ids]
            self.prev_targets[env_ids] = dof_pos[env_ids]
            self.cur_targets[env_ids] = dof_pos[env_ids]
            self.hand_dof_targets[env_ids] = dof_pos[env_ids]
            self.hand.set_joint_position_target(dof_pos[env_ids], env_ids=env_ids)
            self.hand.write_joint_state_to_sim(dof_pos[env_ids], dof_vel[env_ids], env_ids=env_ids)
            self.sim.forward()
            self.scene.update(dt=0.0)
        self.hp_ee_reset_pos[:] = old_reset_pos
        self.hp_ee_reset_quat[:] = old_reset_quat
        self.hp_ee_reset_pending[env_ids] = True

    def _get_reference_pn_expert_action(self) -> torch.Tensor:
        global_step = self.pn_expert_phase_step_count
        attempt_len = 220
        attempt = torch.clamp(global_step // attempt_len, max=2)
        step = global_step - attempt * attempt_len
        prev_attempt = self.pn_expert_attempt_count.clone()
        self.pn_expert_attempt_count[:] = attempt
        self.pn_expert_noise[:] = self.pn_expert_attempt_noise[torch.arange(self.num_envs, device=self.device), attempt]
        latch_grasp = (self.pn_expert_grasp_center.abs().sum(dim=-1) == 0.0) | (attempt != prev_attempt)
        if latch_grasp.any():
            self.pn_expert_grasp_center[latch_grasp] = self.object_pos[latch_grasp] + self.pn_expert_noise[latch_grasp]
            rough_eef_rel = torch.tensor((0.03954, -0.01669), dtype=torch.float32, device=self.device)
            rough_xy = self.pn_expert_grasp_center[latch_grasp, :2] + rough_eef_rel.unsqueeze(0)
            current_xy = self.pn_expert_eef_target[latch_grasp, :2]
            approach_dist = torch.linalg.norm(rough_xy - current_xy, dim=-1)
            self.pn_expert_timing_offset[latch_grasp] = ((approach_dist - 0.035).clamp_min(0.0) / 0.004).round().to(torch.long).clamp(0, 8)

        ref_step = (step - self.pn_expert_timing_offset).clamp_min(0)
        eef_rel, leap_q = self._interpolate_pn_reference(ref_step)
        eef_target = self.pn_expert_grasp_center + eef_rel
        eef_target[:, :2] = eef_target[:, :2] + self._interpolate_pn_xy_stage_offset(ref_step)
        warmup_alpha = (ref_step.to(eef_target.dtype) / 8.0).clamp(0.0, 1.0).unsqueeze(-1)
        eef_target = self.pn_expert_initial_eef_target + warmup_alpha * (eef_target - self.pn_expert_initial_eef_target)
        pick_time_scale = float(getattr(self.cfg, "pn_bd_pick_time_scale", 1.0))
        pregrasp_step = max(1, int(round(42 * pick_time_scale)))
        lift_start = max(pregrasp_step + 1, int(round(float(getattr(self.cfg, "pn_expert_lift_start_step", 100)) * pick_time_scale)))
        lift_step = lift_start + self.pn_expert_timing_offset
        lifting = step >= lift_step
        self.pn_expert_lift_xy_valid[~lifting] = False
        new_lift = lifting & ~self.pn_expert_lift_xy_valid
        if new_lift.any():
            self.pn_expert_lift_xy[new_lift] = self.pn_expert_eef_target[new_lift, :2]
            self.pn_expert_lift_xy_valid[new_lift] = True
        eef_target[lifting, :2] = self.pn_expert_lift_xy[lifting]
        lift_speed_scale = float(getattr(self.cfg, "pn_bd_pick_lift_speed_scale", 1.0))
        lift_delta = (step - lift_step + 1).clamp_min(0).to(eef_target.dtype) * (0.0010 * lift_speed_scale)
        eef_target[:, 2] = eef_target[:, 2] + lift_delta
        if bool(getattr(self.cfg, "pn_pick_freeze_pick_eef_until_lift", False)):
            # In the mass-sensing setup the policy should not spend capacity on
            # the scripted pregrasp horizontal wrist trajectory. Keep XY fixed,
            # but retain the small vertical contact/lift motion needed to grasp.
            eef_target[:, :2] = self.pn_expert_initial_eef_target[:, :2]
        eef_target = self._limit_pn_eef_target_step(eef_target)
        early = ref_step < pregrasp_step
        if early.any():
            alpha = (ref_step[early].to(leap_q.dtype) / float(pregrasp_step)).unsqueeze(-1)
            pregrasp_q = self._pn_pregrasp_leap_q().unsqueeze(0)
            leap_q[early] = self.pn_expert_initial_leap_target[early] + alpha * (pregrasp_q - self.pn_expert_initial_leap_target[early])
        grip_extra_fraction = float(getattr(self.cfg, "pn_bd_pick_grip_extra_fraction", 0.0))
        if grip_extra_fraction > 0.0:
            grip_alpha = ((ref_step.to(leap_q.dtype) - 57.0) / max(float(lift_start) - 57.0, 1.0)).clamp(0.0, 1.0)
            leap_q = leap_q + (grip_alpha * grip_extra_fraction).unsqueeze(-1) * (self._pn_expert_closed_q() - leap_q)
        leap_q = self._apply_pn_fine_alignment_compensation(leap_q, ref_step)

        self.pn_expert_eef_target[:] = eef_target
        self.pn_expert_leap_target[:] = leap_q
        self.pn_expert_phase[:] = torch.where(ref_step < 50, torch.zeros_like(step), torch.where(step < lift_step, torch.full_like(step, 2), torch.full_like(step, 4)))
        success = self._compute_sort_pick_success()
        retry_lift_height = float(getattr(self.cfg, "pn_expert_retry_lift_height", 0.0))
        if retry_lift_height > 0.0:
            grasped = self._compute_sort_grasped()
            retry = lifting & ~grasped & (lift_delta >= retry_lift_height) & (attempt < 2)
            if retry.any():
                next_attempt = (attempt[retry] + 1).clamp(max=2)
                self.pn_expert_phase_step_count[retry] = next_attempt * attempt_len - 1
                self.pn_expert_initial_eef_target[retry] = self.hand.data.body_pos_w[retry, self.hp_ee_body_idx] - self.scene.env_origins[retry]
                self.pn_expert_initial_leap_target[retry] = self.hand.data.joint_pos[retry][:, self.control_dof_indices]
                self.pn_expert_eef_target[retry] = self.pn_expert_initial_eef_target[retry]
                self.pn_expert_leap_target[retry] = self.pn_expert_initial_leap_target[retry]
                self.pn_expert_grasp_center[retry] = 0.0
                self.pn_expert_lift_xy_valid[retry] = False
        self.pn_expert_phase[success] = 5
        self.pn_expert_phase_step_count += 1
        return torch.cat((self.pn_expert_eef_target, self.pn_expert_leap_target), dim=-1)

    def _pn_reference_keyframes(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        times = torch.tensor((0, 17, 33, 42, 57, 73, 90), dtype=torch.float32, device=self.device)
        times = times * float(getattr(self.cfg, "pn_bd_pick_time_scale", 1.0))
        eef_rel = torch.tensor(
            (
                (0.03954, -0.01669, 0.17919),
                (0.03954, -0.01669, 0.17619),
                (0.03954, -0.01669, 0.17119),
                (0.03954, -0.01669, 0.16919),
                (0.03954, -0.01669, 0.16919),
                (0.03954, -0.01669, 0.16919),
                (0.03954, -0.01669, 0.15919),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        leap_q = torch.tensor(
            (
                (-0.00767, 1.07379, -0.47093, 0.01841, 1.55392, -0.06443, -0.82375, 0.04602),
                (-0.02181, 1.13536, -0.41455, 0.04614, 1.53960, -0.05816, -0.73033, 0.07877),
                (-0.05010, 1.25849, -0.30180, 0.10164, 1.51097, -0.04562, -0.54349, 0.14426),
                (-0.05010, 1.25849, -0.30180, 0.10164, 1.51097, -0.04562, -0.54349, 0.14426),
                (-0.07965, 1.38712, -0.18401, 0.15960, 1.48106, -0.03253, -0.34831, 0.21267),
                (-0.08950, 1.43000, -0.14475, 0.17892, 1.47109, -0.02816, -0.28325, 0.23548),
                (-0.12890, 1.60150, 0.01230, 0.25620, 1.43120, -0.01070, -0.02300, 0.32670),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        return times, eef_rel, leap_q

    def _pn_pregrasp_leap_q(self) -> torch.Tensor:
        return torch.tensor((-0.05010, 1.25849, -0.30180, 0.10164, 1.51097, -0.04562, -0.54349, 0.14426), dtype=torch.float32, device=self.device)

    def _pn_expert_open_q(self) -> torch.Tensor:
        q = torch.tensor((-0.3738, 2.0303, 0.4076, 0.3495, 1.1077, 0.4886, -0.2071, 0.9890), dtype=torch.float32, device=self.device)
        return q.unsqueeze(0).repeat(self.num_envs, 1)

    def _pn_expert_home_q(self) -> torch.Tensor:
        q = self.init_joint_values[self.control_dof_indices].to(dtype=torch.float32)
        return q.unsqueeze(0).repeat(self.num_envs, 1)

    def _pn_expert_closed_q(self) -> torch.Tensor:
        q = torch.tensor((-0.7300, 2.1050, -0.1010, 0.8230, 1.0800, 0.6930, -0.4970, 0.9240), dtype=torch.float32, device=self.device)
        return q.unsqueeze(0).repeat(self.num_envs, 1)

    def _compute_pn_expert_leap_targets(self, current_target: torch.Tensor) -> torch.Tensor:
        grasp_target = torch.tensor(
            self.cfg.pn_expert_grasp_joint_target,
            dtype=current_target.dtype,
            device=self.device,
        ).view(1, -1)
        grasp_delta = torch.clamp(
            grasp_target - current_target,
            min=-float(self.cfg.pn_expert_grasp_joint_step),
            max=float(self.cfg.pn_expert_grasp_joint_step),
        )
        tip_pos = self._get_pn_expert_contact_points_env()
        grasp_axis = tip_pos[:, 1, :2] - tip_pos[:, 0, :2]
        grasp_axis = grasp_axis / torch.linalg.vector_norm(grasp_axis, dim=-1, keepdim=True).clamp_min(1e-6)
        side_radius = 0.5 * float(self.cfg.nut_diameter) + float(self.cfg.pn_expert_side_clearance)
        desired = torch.empty_like(tip_pos)
        desired[:, 0, :2] = self.pn_expert_grasp_center[:, :2] - grasp_axis * side_radius
        desired[:, 1, :2] = self.pn_expert_grasp_center[:, :2] + grasp_axis * side_radius
        desired[:, :, 2] = self.object_pos[:, 2:3] + float(self.cfg.pn_expert_contact_z_offset)
        tip_error = desired - tip_pos
        min_clearance_z = float(self.cfg.table_z) + 0.0015
        tip_error[:, :, 2] = torch.maximum(tip_error[:, :, 2], min_clearance_z - tip_pos[:, :, 2])
        error_norm = torch.linalg.vector_norm(tip_error, dim=-1, keepdim=True).clamp_min(1e-6)
        tip_error = tip_error * torch.clamp(float(self.cfg.pn_expert_close_step) / error_norm, max=1.0)
        tip_error = tip_error.reshape(self.num_envs, 6)

        jacobians = self.hand.root_physx_view.get_jacobians()
        jac_list = []
        body_pos = self.hand.data.body_pos_w[:, self.pn_tip_body_indices]
        body_quat = self.hand.data.body_quat_w[:, self.pn_tip_body_indices]
        local = self._get_pn_expert_contact_points_link()
        point_offset = quat_apply(body_quat.reshape(-1, 4), local.reshape(-1, 3)).reshape_as(body_pos)
        for finger_idx, jacobi_idx in enumerate(self.pn_tip_jacobi_indices.tolist()):
            body_jac = jacobians[:, jacobi_idx, 0:6, :][:, :, self.control_dof_indices]
            point_jac = body_jac[:, 0:3, :] - self._skew(point_offset[:, finger_idx]) @ body_jac[:, 3:6, :]
            jac_list.append(point_jac)
        jacobian = torch.cat(jac_list, dim=1)
        jacobian_t = jacobian.transpose(1, 2)
        lambda_sq = float(self.cfg.pn_expert_leap_ik_damping) ** 2
        eye = torch.eye(6, device=self.device, dtype=jacobian.dtype).unsqueeze(0).repeat(self.num_envs, 1, 1)
        delta_q = jacobian_t @ torch.linalg.solve(
            jacobian @ jacobian_t + lambda_sq * eye,
            tip_error.unsqueeze(-1),
        )
        delta_q = float(self.cfg.pn_expert_leap_ik_gain) * delta_q.squeeze(-1)
        delta_q = torch.clamp(
            delta_q,
            min=-float(self.cfg.pn_expert_leap_delta_limit),
            max=float(self.cfg.pn_expert_leap_delta_limit),
        )
        target = current_target + grasp_delta + delta_q
        lower = self.hand_dof_lower_limits[:, self.control_dof_indices]
        upper = self.hand_dof_upper_limits[:, self.control_dof_indices]
        return torch.clamp(target, lower, upper)

    @staticmethod
    def _move_toward(current: torch.Tensor, target: torch.Tensor, max_step: float) -> torch.Tensor:
        delta = target - current
        return current + torch.clamp(delta, min=-max_step, max=max_step)

    def _refresh_pn_expert_state_if_needed(self):
        uninitialized = self.pn_expert_eef_target.abs().sum(dim=-1) == 0.0
        if uninitialized.any():
            env_ids = uninitialized.nonzero(as_tuple=False).squeeze(-1)
            self.pn_expert_eef_target[env_ids] = self.hand.data.body_pos_w[env_ids, self.hp_ee_body_idx] - self.scene.env_origins[env_ids]
            self.pn_expert_eef_base_z[env_ids] = self.pn_expert_eef_target[env_ids, 2]
            self.pn_expert_leap_target[env_ids] = self.hand.data.joint_pos[env_ids][:, self.control_dof_indices]

    def _interpolate_pn_xy_stage_offset(self, step: torch.Tensor) -> torch.Tensor:
        step_f = step.to(dtype=torch.float32)
        rough = self.pn_expert_xy_stage_offsets[:, 0]
        down = self.pn_expert_xy_stage_offsets[:, 1]
        fine = self.pn_expert_xy_stage_offsets[:, 2]
        down_start = float(getattr(self.cfg, "pn_expert_xy_down_start_step", 42.0))
        down_end = float(getattr(self.cfg, "pn_expert_xy_down_end_step", 73.0))
        fine_start = float(getattr(self.cfg, "pn_expert_xy_fine_start_step", 73.0))
        fine_end = float(getattr(self.cfg, "pn_expert_xy_fine_end_step", 90.0))
        down_alpha = ((step_f - down_start) / max(down_end - down_start, 1e-6)).clamp(0.0, 1.0).unsqueeze(-1)
        fine_alpha = ((step_f - fine_start) / max(fine_end - fine_start, 1e-6)).clamp(0.0, 1.0).unsqueeze(-1)
        offset = rough + down_alpha * (down - rough)
        return offset + fine_alpha * (fine - offset)

    def _apply_pn_fine_alignment_compensation(self, leap_q: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        return leap_q

    def _limit_pn_eef_target_step(self, desired_target: torch.Tensor) -> torch.Tensor:
        max_step = float(getattr(self.cfg, "pn_expert_eef_step_limit", 0.004))
        if max_step <= 0.0:
            return desired_target
        delta = desired_target - self.pn_expert_eef_target
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
        return self.pn_expert_eef_target + delta * (max_step / distance).clamp(max=1.0)

    def _interpolate_pn_reference(self, step: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        times, eef_rel, leap_q = self._pn_reference_keyframes()
        step_f = step.to(dtype=torch.float32).clamp(max=float(times[-1]))
        right = torch.searchsorted(times, step_f, right=True).clamp(min=1, max=times.numel() - 1)
        left = right - 1
        denom = (times[right] - times[left]).clamp_min(1e-6)
        alpha = ((step_f - times[left]) / denom).unsqueeze(-1)
        return eef_rel[left] + alpha * (eef_rel[right] - eef_rel[left]), leap_q[left] + alpha * (leap_q[right] - leap_q[left])

    def _get_adaptive_pn_expert_action(self) -> torch.Tensor:
        uninitialized_grasp = self.pn_expert_grasp_center.abs().sum(dim=-1) == 0.0
        if uninitialized_grasp.any():
            self.pn_expert_grasp_center[uninitialized_grasp] = (
                self.object_pos[uninitialized_grasp] + self.pn_expert_noise[uninitialized_grasp]
            )

        tip_pos = self._get_pn_expert_contact_points_env()
        midpoint = tip_pos.mean(dim=1)
        desired_midpoint = self.pn_expert_grasp_center.clone()
        desired_midpoint[:, 2] = self.object_pos[:, 2] + 0.010
        midpoint_error = desired_midpoint - midpoint
        min_tip_z = tip_pos[:, :, 2].min(dim=1).values
        surface_dist = self._get_pn_expert_surface_distances()
        min_clearance_z = float(self.cfg.table_z) + 0.0015
        approach_eef_z_min = 0.052
        grasp_eef_z_min = 0.044
        midpoint_error[:, 2] = torch.maximum(midpoint_error[:, 2], min_clearance_z - min_tip_z)
        midpoint_dist = torch.linalg.vector_norm(midpoint_error[:, :2], dim=-1)
        open_q = self._pn_expert_open_q()
        closed_q = self._pn_expert_closed_q()

        current_q = self.hand.data.joint_pos[:, self.control_dof_indices]
        open_mask = self.pn_expert_phase == 0
        open_done = open_mask & (torch.linalg.vector_norm(current_q - open_q, dim=-1) < 0.040)
        self.pn_expert_phase_step_count[open_mask] += 1
        self.pn_expert_phase[open_done] = 1
        self.pn_expert_phase_step_count[open_done] = 0
        if open_mask.any():
            self.pn_expert_leap_target[open_mask] = self._move_toward(
                self.pn_expert_leap_target[open_mask],
                open_q[open_mask],
                0.025,
            )

        approach_mask = self.pn_expert_phase == 1
        approach_done = approach_mask & (midpoint_dist < float(self.cfg.pn_expert_approach_tol))
        self.pn_expert_phase_step_count[approach_mask] += 1
        self.pn_expert_phase[approach_done] = 2
        self.pn_expert_phase_step_count[approach_done] = 0
        if approach_mask.any():
            self.pn_expert_leap_target[approach_mask] = open_q[approach_mask]
            step = midpoint_error * float(self.cfg.pn_expert_approach_gain)
            step_norm = torch.linalg.vector_norm(step, dim=-1, keepdim=True).clamp_min(1e-6)
            step = step * torch.clamp(float(self.cfg.pn_expert_approach_step_limit) / step_norm, max=1.0)
            self.pn_expert_eef_target[approach_mask] = self.pn_expert_eef_target[approach_mask] + step[approach_mask]
            self.pn_expert_eef_target[approach_mask, 2] = torch.clamp(
                self.pn_expert_eef_target[approach_mask, 2],
                min=approach_eef_z_min,
                max=float(self.cfg.pn_expert_eef_z_max),
            )

        close_mask = self.pn_expert_phase == 2
        if close_mask.any():
            desired_midpoint[:, 2] = self.object_pos[:, 2] + float(self.cfg.pn_expert_contact_z_offset)
            midpoint_error = desired_midpoint - midpoint
            midpoint_error[:, 2] = torch.maximum(midpoint_error[:, 2], min_clearance_z - min_tip_z)
            recenter_step = midpoint_error * float(self.cfg.pn_expert_close_recenter_gain)
            recenter_norm = torch.linalg.vector_norm(recenter_step, dim=-1, keepdim=True).clamp_min(1e-6)
            recenter_step = recenter_step * torch.clamp(
                float(self.cfg.pn_expert_close_recenter_step_limit) / recenter_norm,
                max=1.0,
            )
            self.pn_expert_eef_target[close_mask] = self.pn_expert_eef_target[close_mask] + recenter_step[close_mask]
            self.pn_expert_eef_target[close_mask, 2] = torch.clamp(
                self.pn_expert_eef_target[close_mask, 2],
                min=grasp_eef_z_min,
                max=float(self.cfg.pn_expert_eef_z_max),
            )
            close_steps = 45.0
            alpha = torch.clamp(self.pn_expert_phase_step_count.float().unsqueeze(-1) / close_steps, 0.0, 1.0)
            close_target = open_q + alpha * (closed_q - open_q)
            self.pn_expert_leap_target[close_mask] = self._move_toward(
                self.pn_expert_leap_target[close_mask],
                close_target[close_mask],
                0.035,
            )
            close_done = close_mask & (self.pn_expert_phase_step_count >= int(close_steps))
            self.pn_expert_phase[close_done] = 3
            self.pn_expert_phase_step_count[close_done] = 0

        squeeze_mask = self.pn_expert_phase == 3
        if squeeze_mask.any():
            desired_contact_z = self.object_pos[:, 2] + float(self.cfg.pn_expert_contact_z_offset)
            contact_high_error = desired_contact_z - tip_pos[:, :, 2].mean(dim=1)
            floor_lift_error = min_clearance_z - min_tip_z
            squeeze_z_step = torch.maximum(contact_high_error, floor_lift_error).clamp(min=-0.0020, max=0.0015)
            good_surface = surface_dist.max(dim=1).values < 0.012
            squeeze_z_step = torch.where(good_surface, torch.zeros_like(squeeze_z_step), squeeze_z_step)
            self.pn_expert_eef_target[squeeze_mask, 2] = torch.clamp(
                self.pn_expert_eef_target[squeeze_mask, 2] + squeeze_z_step[squeeze_mask],
                min=grasp_eef_z_min,
                max=float(self.cfg.pn_expert_eef_z_max),
            )
            self.pn_expert_leap_target = self._compute_pn_expert_leap_targets(self.pn_expert_leap_target)

        contact_threshold = float(self.cfg.pn_expert_contact_margin)
        contact = (surface_dist[:, 0] < contact_threshold) & (surface_dist[:, 1] < contact_threshold)
        self.pn_expert_phase_step_count[close_mask] += 1
        self.pn_expert_phase_step_count[squeeze_mask] += 1
        self.pn_expert_overclose_count[squeeze_mask & contact] += 1
        ready_to_lift = (
            squeeze_mask
            & contact
            & (self.pn_expert_overclose_count >= int(self.cfg.pn_expert_overclose_steps))
        )
        if ready_to_lift.any():
            self.pn_expert_phase[ready_to_lift] = 4
            self.pn_expert_phase_step_count[ready_to_lift] = 0
            self.pn_expert_leap_target[ready_to_lift] = self.hand.data.joint_pos[ready_to_lift][
                :, self.control_dof_indices
            ]
            self.pn_expert_lift_target_z[ready_to_lift] = (
                self.pn_expert_eef_target[ready_to_lift, 2] + float(self.cfg.pn_expert_lift_height)
            )

        lift_mask = self.pn_expert_phase == 4
        self.pn_expert_phase_step_count[lift_mask] += 1
        if lift_mask.any():
            lift_step = float(self.cfg.pn_expert_lift_step)
            target_z = self.pn_expert_lift_target_z[lift_mask]
            current_z = self.pn_expert_eef_target[lift_mask, 2]
            self.pn_expert_eef_target[lift_mask, 2] = torch.minimum(current_z + lift_step, target_z)

        success = self._compute_sort_pick_success()
        self.pn_expert_phase[success] = 5
        return torch.cat((self.pn_expert_eef_target, self.pn_expert_leap_target), dim=-1)

    def _move_eef_target_toward(self, target: torch.Tensor, mask: torch.Tensor, max_step: float):
        if not mask.any():
            return
        delta = target - self.pn_expert_eef_target
        distance = torch.linalg.norm(delta, dim=-1, keepdim=True).clamp_min(1e-6)
        step = delta * (max_step / distance).clamp(max=1.0)
        self.pn_expert_eef_target[mask] = self.pn_expert_eef_target[mask] + step[mask]

    def _advance_sort_expert(self, mask: torch.Tensor):
        phase_ids = self._pn_pick_phase_ids()
        sensing_phase = phase_ids["sensing"]
        move_phase = phase_ids["move"]
        lower_phase = phase_ids["lower"]
        release_phase = phase_ids["release"]
        retreat_phase = phase_ids["retreat"]
        done_phase = phase_ids["done"]
        sensing_steps = max(int(getattr(self.cfg, "pn_pick_sensing_hold_steps", 0)), 0)

        target_xy = self._get_sort_target_xy()
        move_target = self.sort_move_eef_target.clone()
        move_target[:, :2] = target_xy + self.sort_hold_eef_rel_xy
        move_target[:, 2] = float(self.cfg.pn_pick_transport_eef_z)
        lower_target = move_target.clone()
        lower_target[:, 2] = float(self.cfg.pn_pick_place_eef_z)
        retreat_target = lower_target.clone()
        retreat_target[:, 2] = float(self.cfg.pn_pick_retreat_eef_z)

        phase = self.sort_phase
        move_mask = mask & (phase == move_phase)
        lower_mask = mask & (phase == lower_phase)
        release_mask = mask & (phase == release_phase)
        retreat_mask = mask & (phase == retreat_phase)

        self._move_eef_target_toward(move_target, move_mask, float(self.cfg.pn_pick_eef_step_limit))
        self._move_eef_target_toward(lower_target, lower_mask, float(self.cfg.pn_pick_lower_eef_step_limit))
        self._move_eef_target_toward(
            lower_target,
            release_mask,
            float(getattr(self.cfg, "pn_pick_release_eef_step_limit", self.cfg.pn_pick_lower_eef_step_limit)),
        )
        if release_mask.any():
            alpha = (self.sort_phase_step_count.to(torch.float32) / max(float(self.cfg.pn_pick_release_steps), 1.0)).clamp(0.0, 1.0).unsqueeze(-1)
            release_q = self._pn_expert_home_q()
            release_fraction = float(getattr(self.cfg, "pn_pick_release_open_fraction", 1.0))
            self.pn_expert_leap_target[release_mask] = (
                self.sort_release_start_q[release_mask]
                + alpha[release_mask]
                * release_fraction
                * (release_q[release_mask] - self.sort_release_start_q[release_mask])
            )
        self._move_eef_target_toward(retreat_target, retreat_mask, float(self.cfg.pn_pick_eef_step_limit))
        if retreat_mask.any():
            alpha = (
                self.sort_phase_step_count.to(torch.float32)
                / max(float(getattr(self.cfg, "pn_pick_return_qpos_steps", 20)), 1.0)
            ).clamp(0.0, 1.0).unsqueeze(-1)
            home_q = self._pn_expert_home_q()
            self.pn_expert_leap_target[retreat_mask] = (
                self.sort_return_start_q[retreat_mask]
                + alpha[retreat_mask] * (home_q[retreat_mask] - self.sort_return_start_q[retreat_mask])
            )
        done_mask = mask & (phase >= done_phase)
        if done_mask.any():
            self.pn_expert_leap_target[done_mask] = self._pn_expert_home_q()[done_mask]

        self.sort_phase_step_count[mask] += 1
        desired = torch.where(
            (phase == move_phase).unsqueeze(-1),
            move_target,
            torch.where(((phase == lower_phase) | (phase == release_phase)).unsqueeze(-1), lower_target, retreat_target),
        )
        arrived = torch.linalg.norm(desired - self.pn_expert_eef_target, dim=-1) < float(self.cfg.pn_pick_waypoint_tolerance)
        moving = mask & ((phase == move_phase) | (phase == lower_phase) | (phase == release_phase) | (phase == retreat_phase))
        arrived_mask = moving & arrived
        self.sort_phase_arrived_count[arrived_mask] += 1
        self.sort_phase_arrived_count[moving & ~arrived] = 0

        hold_steps = int(getattr(self.cfg, "pn_pick_arrive_hold_steps", 4))
        to_move = (phase == sensing_phase) & (self.sort_phase_step_count >= sensing_steps) if sensing_phase > 0 else torch.zeros_like(mask)
        to_lower = (phase == move_phase) & (self.sort_phase_arrived_count >= hold_steps)
        place_xy_error = torch.linalg.vector_norm(self.object_pos[:, :2] - target_xy, dim=-1)
        to_release = (
            (phase == lower_phase)
            & (self.sort_phase_arrived_count >= hold_steps)
            & (place_xy_error < float(getattr(self.cfg, "pn_pick_place_xy_tolerance", 0.006)))
        )
        to_retreat = (
            (phase == release_phase)
            & (self.sort_phase_step_count >= int(self.cfg.pn_pick_release_steps))
            & (place_xy_error < float(getattr(self.cfg, "pn_pick_success_xy_threshold", 0.010)))
        )
        home_q = self._pn_expert_home_q()
        qpos_error = torch.linalg.vector_norm(self.pn_expert_leap_target - home_q, dim=-1)
        qpos_returned = qpos_error < float(getattr(self.cfg, "pn_pick_return_qpos_tolerance", 0.015))
        to_done = (phase == retreat_phase) & (
            (
                (self.sort_phase_arrived_count >= hold_steps)
                & qpos_returned
            )
            | (
                self.sort_phase_step_count
                >= max(
                    int(getattr(self.cfg, "pn_pick_retreat_steps", 20)),
                    int(getattr(self.cfg, "pn_pick_return_qpos_steps", 20)),
                )
            )
        )
        transitions = (
            (to_move, move_phase),
            (to_lower, lower_phase),
            (to_release, release_phase),
            (to_retreat, retreat_phase),
            (to_done, done_phase),
        )
        for phase_mask, next_phase in transitions:
            phase_mask = mask & phase_mask
            if not phase_mask.any():
                continue
            self.sort_phase[phase_mask] = next_phase
            self.sort_phase_step_count[phase_mask] = 0
            self.sort_phase_arrived_count[phase_mask] = 0
            if next_phase == release_phase:
                self.sort_release_start_q[phase_mask] = self.pn_expert_leap_target[phase_mask]
            if next_phase == retreat_phase:
                self.sort_return_start_q[phase_mask] = self.pn_expert_leap_target[phase_mask]
            if next_phase == done_phase:
                self.pn_expert_leap_target[phase_mask] = self._pn_expert_home_q()[phase_mask]
            self.sort_expert_event[phase_mask] = next_phase

    def _apply_pn_pick_kinematic_carry(self, mask: torch.Tensor):
        if not bool(getattr(self.cfg, "pn_pick_kinematic_carry_after_pick", False)):
            return
        if not hasattr(self, "sort_phase") or not mask.any():
            return

        phase_ids = self._pn_pick_phase_ids()
        sensing_phase = phase_ids["sensing"]
        move_phase = phase_ids["move"]
        lower_phase = phase_ids["lower"]
        release_phase = phase_ids["release"]

        phase = self.sort_phase
        active = mask & (phase > 0)
        if not active.any():
            return

        target_xy = self._get_sort_target_xy()
        table_center_z = float(self.cfg.table_z) + float(
            getattr(self.cfg, "nut_reset_center_z", 0.5 * self.cfg.nut_height)
        )
        carry_z = table_center_z + float(getattr(self.cfg, "pn_pick_kinematic_carry_height", 0.020))
        place_steps = max(int(getattr(self.cfg, "pn_pick_kinematic_place_steps", 8)), 1)

        pos = self.object_pos.clone()

        carry_mask = active & ((phase == move_phase) | ((sensing_phase > 0) & (phase == sensing_phase)))
        if carry_mask.any():
            pos[carry_mask, :2] = self.pn_expert_eef_target[carry_mask, :2] - self.sort_hold_eef_rel_xy[carry_mask]
            pos[carry_mask, 2] = carry_z

        lower_mask = active & (phase == lower_phase)
        if lower_mask.any():
            alpha = (
                self.sort_phase_step_count.to(torch.float32) / float(place_steps)
            ).clamp(0.0, 1.0)
            pos[lower_mask, :2] = target_xy[lower_mask]
            pos[lower_mask, 2] = carry_z + alpha[lower_mask] * (table_center_z - carry_z)

        placed_mask = active & (phase >= release_phase)
        if placed_mask.any():
            pos[placed_mask, :2] = target_xy[placed_mask]
            pos[placed_mask, 2] = table_center_z

        env_ids = active.nonzero(as_tuple=False).squeeze(-1)
        if env_ids.numel() == 0:
            return
        pose_w = torch.cat((pos[env_ids] + self.scene.env_origins[env_ids], self.object_rot[env_ids]), dim=-1)
        self.object.write_root_pose_to_sim(pose_w, env_ids)
        self.object.write_root_velocity_to_sim(
            torch.zeros((env_ids.numel(), 6), dtype=torch.float32, device=self.device),
            env_ids,
        )
        self.object_pos[env_ids] = pos[env_ids]

    def get_expert_action(self) -> torch.Tensor:
        self._pn_expert_action_active = False
        self._compute_intermediate_values()
        self._refresh_pn_expert_state_if_needed()
        if hasattr(self, "sort_expert_event"):
            self.sort_expert_event.zero_()

        if self.compute_expert_record_success().all():
            self.pn_expert_phase[:] = 5
            self._pn_expert_action_active = True
            return torch.cat((self.pn_expert_eef_target, self.pn_expert_leap_target), dim=-1)

        self._pn_expert_action_active = True
        if bool(getattr(self.cfg, "pn_pick_use_adaptive_pick_expert", False)):
            return self._get_adaptive_pn_expert_action()
        return self._get_reference_pn_expert_action()

    def _apply_action(self) -> None:
        if self.cfg.replay_mode:
            current_joints = self.actions[:, : self.num_hand_dofs]
            self.hand.write_joint_state_to_sim(position=current_joints, velocity=torch.zeros_like(current_joints))
            nut_pose = self.actions[:, self.num_hand_dofs : self.num_hand_dofs + 7].clone()
            nut_pose[:, :3] += self.scene.env_origins
            self.object.write_root_pose_to_sim(nut_pose)
            self.object.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device))
            return

        self.cur_targets[:] = self.init_joint_values.clone()
        eef_target_env = self.actions[:, :3]
        if self.hp_ee_reset_pending.any():
            env_ids = self.hp_ee_reset_pending.nonzero(as_tuple=False).squeeze(-1)
            self.hp_ee_reset_pos[env_ids] = self.hand.data.body_pos_w[env_ids, self.hp_ee_body_idx] - self.scene.env_origins[env_ids]
            self.hp_ee_reset_quat[env_ids] = self.hand.data.body_quat_w[env_ids, self.hp_ee_body_idx]
            self.hp_ee_reset_pending[env_ids] = False
        arm_targets = self._compute_pn_arm_joint_targets_absolute(eef_target_env)
        expert_action_active = bool(getattr(self, "_pn_expert_action_active", False))
        if expert_action_active and hasattr(self, "pn_expert_phase_step_count"):
            hold_arm = self.pn_expert_phase_step_count <= 2
            if hold_arm.any():
                arm_targets[hold_arm] = self.hand.data.joint_pos[hold_arm][:, self.hp_arm_joint_indices]
        self.cur_targets[:, self.hp_arm_joint_indices] = arm_targets
        self.cur_targets[:, self.control_dof_indices] = self.actions[:, 3:]
        self.hand.set_joint_position_target(self.cur_targets)
        if expert_action_active and hasattr(self, "pn_expert_phase_step_count"):
            stabilize = self.pn_expert_phase_step_count <= 2
            if stabilize.any():
                self.hand.write_joint_state_to_sim(
                    self.cur_targets[stabilize],
                    torch.zeros_like(self.cur_targets[stabilize]),
                    env_ids=stabilize.nonzero(as_tuple=False).squeeze(-1),
                )

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        task_success = self._compute_task_success()
        expert_pick = self._compute_sort_pick_success()
        nut_top_z = self.object_pos[:, 2] + 0.5 * float(self.cfg.nut_height)
        lift_reward = torch.clamp(nut_top_z / max(float(self.cfg.pn_success_height), 1.0e-6), 0.0, 1.0)
        reward = 0.2 * expert_pick.float() + 0.3 * lift_reward + task_success.float()
        self.extras["pn/pick_rate"] = float(expert_pick.float().mean())
        self.extras["pn/success_rate"] = float(task_success.float().mean())
        self.extras["pn/nut_top_z"] = float(nut_top_z.mean())
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        success = self._compute_task_success()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        dropped_after_pick = (
            (self.object_pos[:, 2] < float(getattr(self.cfg, "pn_pick_drop_height", 0.010)))
            & self.sort_picked
            & bool(getattr(self.cfg, "pn_pick_reset_on_drop", False))
        )
        success_done = success if getattr(self.cfg, "reset_on_success", True) else torch.zeros_like(success)
        return success_done | dropped_after_pick, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if torch.is_inference_mode_enabled():
            with torch.inference_mode(False):
                return self._reset_idx(env_ids)
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        internal_double_reset = bool(getattr(self, "_pn_pick_internal_double_reset", False))
        super(FingerEyeLabEnv, self)._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if not hasattr(self, "sort_reset_counter"):
            self.sort_reset_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if not internal_double_reset:
            self.sort_reset_counter[env_ids] += 1
        random_env_ids = env_ids[1:]

        if self.rand_cfg.enable_all or self.rand_cfg.random_lighting:
            self._randomize_env_lighting(random_env_ids)
        if self.rand_cfg.enable_all or self.rand_cfg.random_object_color:
            self._randomize_object_visual_color(random_env_ids)
        if self.rand_cfg.enable_all or self.rand_cfg.random_background:
            self._randomize_env_skybox(env_ids)
            self._randomize_env_floor_color(random_env_ids, base_gray=0.7, delta=0.3)

        self._reset_nut_asset(env_ids)
        self._reset_hand(env_ids)
        self.scene.update(dt=0.0)
        self.successes[env_ids] = 0
        self._compute_intermediate_values(env_ids)
        self._reset_pn_expert_state(env_ids)
        self._reset_sort_expert_state(env_ids)
        self._apply_pn_bd_diverse_initial_wrist_pose(env_ids)
        self._reset_pn_expert_state(env_ids, resample_noise=False)
        self._reset_sort_expert_state(env_ids)

        idle_action = torch.cat(
            (
                self.hand.data.body_pos_w[:, self.hp_ee_body_idx] - self.scene.env_origins,
                self.hand.data.joint_pos[:, self.control_dof_indices],
            ),
            dim=-1,
        )
        old_actions = self.actions.clone() if hasattr(self, "actions") else None
        if hasattr(self, "actions") and self.actions.is_inference():
            self.actions = self.actions.clone()
        for _ in range(max(1, int(getattr(self.cfg, "reset_stabilization_steps", 1)))):
            if not self.cfg.replay_mode:
                self.actions[:] = idle_action
                self._apply_action()
                self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=0.0)
        self._enforce_nut_reset_yaw(env_ids)
        self.scene.update(dt=0.0)
        for _ in range(max(0, int(getattr(self.cfg, "reset_post_yaw_stabilization_steps", 0)))):
            if not self.cfg.replay_mode:
                self.actions[:] = idle_action
                self._apply_action()
                self.scene.write_data_to_sim()
            self.sim.step()
            self.scene.update(dt=0.0)
        if old_actions is not None:
            self.actions[:] = old_actions
        self.sim.forward()
        self.scene.update(dt=0.0)
        self.hp_ee_reset_pos[env_ids] = self.hand.data.body_pos_w[env_ids, self.hp_ee_body_idx] - self.scene.env_origins[env_ids]
        self.hp_ee_reset_quat[env_ids] = self.hand.data.body_quat_w[env_ids, self.hp_ee_body_idx]
        self.hp_ee_reset_pending[env_ids] = False
        self._reset_pn_expert_state(env_ids, resample_noise=False)
        self._reset_sort_expert_state(env_ids)
        self.sim.forward()
        self.scene.update(dt=0.0)
        if not internal_double_reset:
            self._pn_pick_internal_double_reset = True
            try:
                self._reset_idx(env_ids)
            finally:
                self._pn_pick_internal_double_reset = False
        self._compute_intermediate_values(env_ids)

    def _compute_intermediate_values(self, env_ids: Sequence[int] | None = None):
        FingerEyeLabEnv._compute_intermediate_values(self, env_ids)
        if not hasattr(self, "sort_picked"):
            self.sort_picked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        if hasattr(self, "sort_phase"):
            if env_ids is None:
                carry_mask = self.sort_phase > 0
            else:
                carry_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                carry_mask[env_ids] = self.sort_phase[env_ids] > 0
            self._apply_pn_pick_kinematic_carry(carry_mask)
        pick_success = self._compute_sort_pick_success()
        self.sort_picked |= pick_success
        target_xy = self._get_sort_target_xy()
        self.pn_pick_xy_error = torch.linalg.vector_norm(self.object_pos[:, :2] - target_xy, dim=-1)
        table_center_z = float(self.cfg.table_z) + float(getattr(self.cfg, "nut_reset_center_z", 0.5 * self.cfg.nut_height))
        self.pn_pick_z_error = torch.abs(self.object_pos[:, 2] - table_center_z)

    def _get_observations(self) -> dict:
        obs = FingerEyeLabEnv._get_observations(self)
        hp_eef_pos = self.hand.data.body_pos_w[:, self.hp_ee_body_idx] - self.scene.env_origins
        hp_eef_quat = self.hand.data.body_quat_w[:, self.hp_ee_body_idx]
        between_tips, near_grasp_line, grasp_line_xy_error = self._compute_sort_between_fingertips()
        grasped = self._compute_sort_grasped()
        success = self._compute_task_success()
        expert_success = self.compute_expert_record_success()

        obs["hp_eef_pose"] = torch.cat((hp_eef_pos, hp_eef_quat), dim=-1)
        obs["nut_pose"] = torch.cat((self.object_pos, self.object_rot), dim=-1)
        obs["pn_success"] = success[:, None].float()
        obs["pn_expert_success"] = expert_success[:, None].float()
        obs["pn_pick_phase"] = self.sort_phase[:, None].float()
        obs["pn_pick_phase_step_count"] = self.sort_phase_step_count[:, None].float()
        obs["pn_pick_picked"] = self.sort_picked[:, None].float()
        obs["pn_pick_grasped"] = grasped[:, None].float()
        obs["pn_pick_between_tips"] = between_tips[:, None].float()
        obs["pn_pick_near_grasp_line"] = near_grasp_line[:, None].float()
        obs["pn_pick_grasp_line_xy_error"] = grasp_line_xy_error[:, None]
        obs["pn_pick_target_xy"] = self._get_sort_target_xy()
        obs["pn_pick_xy_error"] = self.pn_pick_xy_error[:, None]
        obs["pn_pick_z_error"] = self.pn_pick_z_error[:, None]
        obs["pn_pick_expert_event"] = self.sort_expert_event[:, None].float()
        obs["pn_pick_sensing_active"] = (
            self.sort_phase == self._pn_pick_phase_ids()["sensing"]
        )[:, None].float()
        obs["task_success"] = success[:, None].float()

        # Backward-compatible aliases used by existing visualization and data scripts.
        obs["pn_bd_phase"] = obs["pn_pick_phase"]
        obs["pn_bd_phase_step_count"] = obs["pn_pick_phase_step_count"]
        obs["pn_bd_picked"] = obs["pn_pick_picked"]
        obs["pn_bd_grasped"] = obs["pn_pick_grasped"]
        obs["pn_bd_between_tips"] = obs["pn_pick_between_tips"]
        obs["pn_bd_near_grasp_line"] = obs["pn_pick_near_grasp_line"]
        obs["pn_bd_grasp_line_xy_error"] = obs["pn_pick_grasp_line_xy_error"]
        obs["pn_bd_reset_counter"] = self.sort_reset_counter[:, None].float()
        obs["pn_bd_expert_event"] = obs["pn_pick_expert_event"]
        obs["pn_bd_inserted"] = success[:, None].float()
        obs["pn_bd_xy_error"] = torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        obs["pn_bd_z_error"] = (
            self.object_pos[:, 2] + 0.5 * float(self.cfg.nut_height) - float(self.cfg.pn_success_height)
        )[:, None]
        obs["pn_bd_nut_rel_bolt"] = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        obs["pn_only_target_xy"] = obs["pn_pick_target_xy"]
        obs["pn_only_success"] = success[:, None].float()
        obs["coin_pose"] = obs["nut_pose"]
        obs["pos_of_coin"] = self.object_pos
        obs["coin_z_axis"] = quat_to_M6(self.object_rot)
        return obs
