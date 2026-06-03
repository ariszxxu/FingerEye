import math
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.sensors import TiledCamera
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sim.utils import bind_physics_material, bind_visual_material, get_current_stage
from isaaclab.utils.math import quat_apply, sample_uniform
from pxr import Gf, Usd, UsdGeom, UsdShade

from .env_tools import ensure_floor_uv, quat_to_M6
from .fingereye_cih_env_cfg import FingerEyeCIHLabEnvCfg, _cih_physics_material
from .fingereye_env import FingerEyeLabEnv


class FingerEyeCIHLabEnv(FingerEyeLabEnv):
    """Final cylinder-in-hole FingerEye task."""

    cfg: FingerEyeCIHLabEnvCfg

    def _setup_scene(self):
        self.hand = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(self.cfg.object_cfg)
        self.plate = RigidObject(self.cfg.plate_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.setup_gray_ground()

        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.hand
        self.scene.rigid_objects["object"] = self.object
        self.scene.rigid_objects["cylinder"] = self.object
        self.scene.rigid_objects["plate"] = self.plate
        self.scene.rigid_objects["hole_plate"] = self.plate

        stage = get_current_stage()
        for env_id in range(self.num_envs):
            ensure_floor_uv(stage, f"/World/envs/env_{env_id}/Background/floor")
            self._setup_range_markers(stage, env_id)
            if bool(getattr(self.cfg, "show_cih_hole_bottom", True)):
                self._setup_hole_bottom(stage, env_id)
            if bool(getattr(self.cfg, "show_cih_hole_ring", True)):
                self._setup_hole_ring(stage, env_id)
            self._set_plate_ground_material(stage, env_id)
            self._bind_cih_physics_material(stage, env_id)

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
                camera_cfg = self._camera_configs.get(name)
                if camera_cfg is None:
                    continue
                print(f"[INFO] Spawning Camera: {name} ({camera_cfg.width}x{camera_cfg.height})")
                if camera_cfg.spawn is None:
                    self._clone_camera_with_standard_xform_ops(stage, camera_cfg)
                sensor = TiledCamera(camera_cfg)
                self.scene.sensors[name] = sensor
                self._active_cameras[name] = sensor
                self._register_camera_body_pose_source(name, camera_cfg, stage)

        self._cih_visual_targets = []
        self._cih_success_visual_state = [None] * self.num_envs
        for env_id in range(self.num_envs):
            self._cih_visual_targets.append(self._collect_cylinder_visual_targets(stage, env_id))
        self._set_cylinder_success_visual([False] * self.num_envs, force=True)

    def _setup_range_markers(self, stage, env_id: int):
        env_path = f"/World/envs/env_{env_id}"
        z = float(self.cfg.table_z) + float(self.cfg.plate_thickness) + 0.0012
        self._make_range_box(
            stage,
            f"{env_path}/Debug/cylinder_left_pick_range",
            float(self.cfg.marker_nut_left_x_min),
            float(self.cfg.marker_nut_left_x_max),
            float(self.cfg.marker_nut_left_y_min),
            float(self.cfg.marker_nut_left_y_max),
            z,
            Gf.Vec3f(1.0, 0.0, 0.0),
        )
        self._make_range_box(
            stage,
            f"{env_path}/Debug/cylinder_right_pick_range",
            float(self.cfg.marker_nut_right_x_min),
            float(self.cfg.marker_nut_right_x_max),
            float(self.cfg.marker_nut_right_y_min),
            float(self.cfg.marker_nut_right_y_max),
            z,
            Gf.Vec3f(1.0, 0.0, 0.0),
        )
        self._make_range_box(
            stage,
            f"{env_path}/Debug/hole_place_range",
            float(self.cfg.marker_hole_x_min),
            float(self.cfg.marker_hole_x_max),
            float(self.cfg.marker_hole_y_min),
            float(self.cfg.marker_hole_y_max),
            z,
            Gf.Vec3f(0.0, 1.0, 0.0),
        )

    def _make_range_box(self, stage, prim_path: str, x_min: float, x_max: float, y_min: float, y_max: float, z: float, color):
        thickness = 0.0012
        height = 0.0006
        x_mid = 0.5 * (x_min + x_max)
        y_mid = 0.5 * (y_min + y_max)
        x_len = x_max - x_min
        y_len = y_max - y_min
        bars = {
            "x_min": ((x_min, y_mid, z), (thickness, y_len + thickness, height)),
            "x_max": ((x_max, y_mid, z), (thickness, y_len + thickness, height)),
            "y_min": ((x_mid, y_min, z), (x_len + thickness, thickness, height)),
            "y_max": ((x_mid, y_max, z), (x_len + thickness, thickness, height)),
        }
        for name, (translation, scale) in bars.items():
            cube = UsdGeom.Cube.Define(stage, f"{prim_path}/{name}")
            cube.CreateSizeAttr(1.0)
            prim = cube.GetPrim()
            xform = UsdGeom.Xformable(prim)
            xform.ClearXformOpOrder()
            xform.AddTranslateOp().Set(Gf.Vec3d(*translation))
            xform.AddScaleOp().Set(Gf.Vec3f(*scale))
            gprim = UsdGeom.Gprim(prim)
            gprim.CreateDisplayColorAttr().Set([color])
            gprim.CreateDisplayOpacityAttr().Set([1.0])

    def _setup_hole_bottom(self, stage, env_id: int):
        radius = float(self.cfg.hole_radius) * float(getattr(self.cfg, "cih_hole_bottom_radius_scale", 0.92))
        z = -0.5 * float(self.cfg.plate_thickness) + float(getattr(self.cfg, "cih_hole_bottom_z_offset", 0.00005))
        segments = max(12, int(getattr(self.cfg, "cih_hole_ring_segments", 96)))
        points = [Gf.Vec3f(0.0, 0.0, z)]
        for i in range(segments):
            angle = 2.0 * math.pi * i / segments
            points.append(Gf.Vec3f(radius * math.cos(angle), radius * math.sin(angle), z))
        face_counts = [3] * segments
        face_indices = []
        for i in range(segments):
            face_indices.extend([0, 1 + i, 1 + ((i + 1) % segments)])

        mesh = UsdGeom.Mesh.Define(stage, f"/World/envs/env_{env_id}/HolePlate/VisualWhiteHoleBottom")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(face_indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        color = getattr(self.cfg, "cih_hole_bottom_color", (1.0, 1.0, 1.0))
        opacity = float(getattr(self.cfg, "cih_hole_bottom_opacity", 1.0))
        gprim = UsdGeom.Gprim(mesh.GetPrim())
        gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        gprim.CreateDisplayOpacityAttr().Set([opacity])
        self._bind_preview_surface_color(
            stage,
            mesh.GetPrim(),
            f"/World/envs/env_{env_id}/Looks/CIHHoleBottomWhite",
            color,
        )

    def _setup_hole_ring(self, stage, env_id: int):
        inner_radius = float(self.cfg.hole_radius)
        outer_radius = inner_radius + float(self.cfg.cylinder_radius)
        z = 0.5 * float(self.cfg.plate_thickness) + float(getattr(self.cfg, "cih_hole_ring_z_offset", 0.00008))
        segments = max(12, int(getattr(self.cfg, "cih_hole_ring_segments", 96)))
        points = []
        for radius in (outer_radius, inner_radius):
            for i in range(segments):
                angle = 2.0 * math.pi * i / segments
                points.append(Gf.Vec3f(radius * math.cos(angle), radius * math.sin(angle), z))
        face_counts = [4] * segments
        face_indices = []
        for i in range(segments):
            face_indices.extend([i, (i + 1) % segments, segments + ((i + 1) % segments), segments + i])

        mesh = UsdGeom.Mesh.Define(stage, f"/World/envs/env_{env_id}/HolePlate/VisualBlueHoleRing")
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(face_counts)
        mesh.CreateFaceVertexIndicesAttr(face_indices)
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDoubleSidedAttr(True)
        color = getattr(self.cfg, "cih_hole_ring_color", (0.0, 0.18, 1.0))
        opacity = float(getattr(self.cfg, "cih_hole_ring_opacity", 0.95))
        gprim = UsdGeom.Gprim(mesh.GetPrim())
        gprim.CreateDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
        gprim.CreateDisplayOpacityAttr().Set([opacity])
        self._bind_preview_surface_color(
            stage,
            mesh.GetPrim(),
            f"/World/envs/env_{env_id}/Looks/CIHHoleRingBlue",
            color,
        )

    def _set_plate_ground_material(self, stage, env_id: int):
        plate_root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/HolePlate")
        if not plate_root.IsValid():
            return
        material_prim = stage.GetPrimAtPath("/World/ground/Looks/OverwriteGray")
        ground_material = UsdShade.Material(material_prim) if material_prim.IsValid() else None
        color = Gf.Vec3f(0.3, 0.3, 0.3)
        for prim in Usd.PrimRange(plate_root):
            if prim.IsA(UsdGeom.Gprim):
                prim_name = prim.GetName()
                if prim_name in {"VisualBlueHoleRing", "VisualWhiteHoleBottom"}:
                    continue
                gprim = UsdGeom.Gprim(prim)
                color_attr = gprim.GetDisplayColorAttr()
                colors = color_attr.Get() if color_attr else None
                if colors is not None and len(colors) > 1:
                    continue
                is_hole_wall = prim_name == "hole_wall"
                if ground_material and not is_hole_wall:
                    UsdShade.MaterialBindingAPI(prim).Bind(ground_material)
                if not color_attr:
                    color_attr = gprim.CreateDisplayColorAttr()
                color_attr.Set([Gf.Vec3f(0.0, 0.25, 1.0)] if is_hole_wall else [color])

    def _bind_cih_physics_material(self, stage, env_id: int):
        material_path = f"/World/envs/env_{env_id}/CIHPhysicsMaterial"
        material_cfg = _cih_physics_material()
        material_cfg.func(material_path, material_cfg)
        for prim_path in (f"/World/envs/env_{env_id}/Cylinder", f"/World/envs/env_{env_id}/HolePlate"):
            if stage.GetPrimAtPath(prim_path).IsValid():
                bind_physics_material(prim_path, material_path, stage=stage)

    def _apply_action(self) -> None:
        if self.cfg.replay_mode:
            current_joints = self.actions[:, : self.num_hand_dofs]
            self.hand.write_joint_state_to_sim(position=current_joints, velocity=torch.zeros_like(current_joints))
            cylinder_pose = self.actions[:, self.num_hand_dofs : self.num_hand_dofs + 7].clone()
            cylinder_pose[:, :3] += self.scene.env_origins
            self.object.write_root_pose_to_sim(cylinder_pose)
            self.object.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device))
            if self.actions.shape[1] >= self.num_hand_dofs + 14:
                plate_pose = self.actions[:, self.num_hand_dofs + 7 : self.num_hand_dofs + 14].clone()
                plate_pose[:, :3] += self.scene.env_origins
                self.plate.write_root_pose_to_sim(plate_pose)
                self.plate.write_root_velocity_to_sim(
                    torch.zeros(self.num_envs, 6, dtype=torch.float32, device=self.device)
                )
        else:
            self.cur_targets[:] = self.init_joint_values.clone()
            self.cur_targets[:, self.control_dof_indices] = self.actions
            self.hand.set_joint_position_target(self.cur_targets)

    def _get_observations(self) -> dict:
        self._compute_intermediate_values()
        success = self._compute_cih_success()
        if self._success_visual_should_run():
            changed = self._set_cylinder_success_visual(success)
            if changed and self.cfg.enable_cameras and self.sim.has_rtx_sensors():
                self.sim.render()
                self.scene.update(dt=0.0)

        obs = FingerEyeLabEnv._get_observations(self)

        cylinder_pose = torch.cat((self.object_pos, self.object_rot), dim=-1)
        plate_pose = torch.cat((self.plate_pos, self.plate_rot), dim=-1)
        cylinder_rel_hole = self.object_pos - self.hole_pos
        success = self._compute_cih_success()

        obs["cylinder_pose"] = cylinder_pose
        obs["plate_pose"] = plate_pose
        obs["hole_pose"] = plate_pose
        obs["cih_cylinder_pose"] = cylinder_pose
        obs["cih_hole_pos"] = self.hole_pos
        obs["cih_cylinder_pos_rel_hole"] = cylinder_rel_hole
        obs["cih_cylinder_rel_hole"] = cylinder_rel_hole
        obs["cih_xy_error"] = self.cih_xy_error[:, None]
        obs["cih_z_error"] = self.cih_z_error[:, None]
        obs["cih_upright_cos"] = self.cih_upright_cos[:, None]
        obs["cih_all_qpos"] = self.hand_dof_pos[:, self.control_dof_indices]
        obs["task_success"] = success[:, None].float()
        obs["cylinder_z_axis"] = quat_to_M6(self.object_rot)
        return obs

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        success = self._compute_cih_success()
        xy_reward = torch.exp(-self.cih_xy_error / max(float(self.cfg.cih_success_xy_threshold), 1e-6))
        z_reward = torch.exp(-torch.abs(self.cih_z_error) / max(float(self.cfg.cih_success_z_threshold), 1e-6))
        reward = 0.2 * xy_reward + 0.2 * z_reward + success.float()
        self.extras["cih/xy_error"] = float(self.cih_xy_error.mean())
        self.extras["cih/z_error"] = float(self.cih_z_error.mean())
        self.extras["cih/upright_cos"] = float(self.cih_upright_cos.mean())
        self.extras["cih/success_rate"] = float(success.float().mean())
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        success = self._compute_cih_success()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = success if getattr(self.cfg, "reset_on_success", False) else torch.zeros_like(time_out)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super(FingerEyeLabEnv, self)._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        random_env_ids = env_ids[1:]

        if self.rand_cfg.enable_all or self.rand_cfg.random_lighting:
            self._randomize_env_lighting(random_env_ids)
        if self.rand_cfg.enable_all or self.rand_cfg.random_object_color:
            self._randomize_object_visual_color(random_env_ids)
        if self.rand_cfg.enable_all or self.rand_cfg.random_background:
            self._randomize_env_skybox(env_ids)
            self._randomize_env_floor_color(random_env_ids, base_gray=0.7, delta=0.3)

        self._reset_cih_assets(env_ids)
        self._reset_hand(env_ids)
        self.successes[env_ids] = 0
        self._set_cylinder_success_visual([False] * len(env_ids), env_ids=env_ids, force=True)
        self._compute_intermediate_values(env_ids)
        self.sim.step()

    def _reset_cih_assets(self, env_ids: torch.Tensor):
        n = len(env_ids)
        plate_x = sample_uniform(self.cfg.hole_x_min, self.cfg.hole_x_max, (n,), device=self.device)
        plate_y = sample_uniform(self.cfg.hole_y_min, self.cfg.hole_y_max, (n,), device=self.device)
        plate_state = self.plate.data.default_root_state.clone()[env_ids]
        plate_state[:, 0] = plate_x
        plate_state[:, 1] = plate_y
        plate_state[:, 2] = float(self.cfg.table_z) + 0.5 * float(self.cfg.plate_thickness)
        plate_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        plate_state[:, 0:3] += self.scene.env_origins[env_ids]
        plate_state[:, 7:] = 0.0

        choose_right = torch.rand(n, device=self.device) > 0.5
        left_x = sample_uniform(self.cfg.cylinder_left_x_min, self.cfg.cylinder_left_x_max, (n,), device=self.device)
        right_x = sample_uniform(self.cfg.cylinder_right_x_min, self.cfg.cylinder_right_x_max, (n,), device=self.device)
        left_y = sample_uniform(self.cfg.cylinder_left_y_min, self.cfg.cylinder_left_y_max, (n,), device=self.device)
        right_y = sample_uniform(self.cfg.cylinder_right_y_min, self.cfg.cylinder_right_y_max, (n,), device=self.device)
        cylinder_state = self.object.data.default_root_state.clone()[env_ids]
        cylinder_state[:, 0] = torch.where(choose_right, right_x, left_x)
        cylinder_state[:, 1] = torch.where(choose_right, right_y, left_y)
        cylinder_state[:, 2] = (
            float(self.cfg.table_z) + float(self.cfg.plate_thickness) + 0.5 * float(self.cfg.cylinder_height) + 0.001
        )
        yaw = sample_uniform(self.cfg.cylinder_yaw_min, self.cfg.cylinder_yaw_max, (n,), device=self.device)
        cylinder_state[:, 3] = torch.cos(0.5 * yaw)
        cylinder_state[:, 4] = 0.0
        cylinder_state[:, 5] = 0.0
        cylinder_state[:, 6] = torch.sin(0.5 * yaw)
        cylinder_state[:, 0:3] += self.scene.env_origins[env_ids]
        cylinder_state[:, 7:] = 0.0

        self.plate.write_root_pose_to_sim(plate_state[:, :7], env_ids)
        self.plate.write_root_velocity_to_sim(plate_state[:, 7:], env_ids)
        self.object.write_root_pose_to_sim(cylinder_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(cylinder_state[:, 7:], env_ids)

    def _reset_hand(self, env_ids: torch.Tensor):
        dof_pos = self.init_joint_values.unsqueeze(0).repeat(len(env_ids), 1)
        dof_vel = self.hand.data.default_joint_vel[env_ids]
        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos
        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

    def _compute_intermediate_values(self, env_ids: Sequence[int] | None = None):
        FingerEyeLabEnv._compute_intermediate_values(self, env_ids)
        self.plate_pos = self.plate.data.root_pos_w - self.scene.env_origins
        self.plate_rot = self.plate.data.root_quat_w
        self.hole_pos = self.plate_pos.clone()
        self.hole_rot = self.plate_rot
        self.hole_pos[:, 2] = float(self.cfg.table_z) + 0.5 * float(self.cfg.cylinder_height)
        self._compute_cih_success()

    def _compute_cih_success(self) -> torch.Tensor:
        xy_delta = self.object_pos[:, :2] - self.hole_pos[:, :2]
        self.cih_xy_error = torch.linalg.vector_norm(xy_delta, dim=-1)
        self.cih_z_error = self.object_pos[:, 2] - self.hole_pos[:, 2]
        cylinder_axis = quat_apply(
            self.object_rot,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs, 1),
        )
        self.cih_upright_cos = torch.abs(cylinder_axis[:, 2])
        return (
            (self.cih_xy_error < float(self.cfg.cih_success_xy_threshold))
            & (torch.abs(self.cih_z_error) < float(self.cfg.cih_success_z_threshold))
            & (self.cih_upright_cos > float(self.cfg.cih_success_upright_cos))
        )

    def _success_visual_should_run(self) -> bool:
        if not bool(getattr(self.cfg, "cih_success_visual_enabled", True)):
            return False
        replay_only = bool(getattr(self.cfg, "cih_success_visual_replay_only", False))
        return (not replay_only) or bool(getattr(self.cfg, "replay_mode", False))

    def _bind_preview_surface_color(self, stage, prim, material_path: str, color) -> None:
        color_tuple = (float(color[0]), float(color[1]), float(color[2]))
        looks_path = material_path.rsplit("/", 1)[0]
        if not stage.GetPrimAtPath(looks_path).IsValid():
            UsdGeom.Scope.Define(stage, looks_path)
        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim.IsValid():
            material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color_tuple)
            material_cfg.func(material_path, material_cfg)
            material_prim = stage.GetPrimAtPath(material_path)
        else:
            shader_prim = stage.GetPrimAtPath(f"{material_path}/Shader")
            attr = shader_prim.GetAttribute("inputs:diffuseColor") if shader_prim.IsValid() else None
            if attr:
                attr.Set(Gf.Vec3f(*color_tuple))
        if material_prim.IsValid():
            bind_visual_material(prim.GetPath().pathString, material_path, stage=stage, stronger_than_descendants=True)

    def _compute_replay_success_from_actions(self) -> torch.Tensor:
        if self.actions.shape[1] < self.num_hand_dofs + 14:
            self._compute_intermediate_values()
            return self._compute_cih_success()
        cylinder_pose = self.actions[:, self.num_hand_dofs : self.num_hand_dofs + 7]
        plate_pose = self.actions[:, self.num_hand_dofs + 7 : self.num_hand_dofs + 14]
        cylinder_pos = cylinder_pose[:, :3]
        cylinder_rot = cylinder_pose[:, 3:7]
        hole_pos = plate_pose[:, :3].clone()
        hole_pos[:, 2] = float(self.cfg.table_z) + 0.5 * float(self.cfg.cylinder_height)
        xy_error = torch.linalg.vector_norm(cylinder_pos[:, :2] - hole_pos[:, :2], dim=-1)
        z_error = cylinder_pos[:, 2] - hole_pos[:, 2]
        cylinder_axis = quat_apply(
            cylinder_rot,
            torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=cylinder_rot.dtype).repeat(self.num_envs, 1),
        )
        upright_cos = torch.abs(cylinder_axis[:, 2])
        return (
            (xy_error < float(self.cfg.cih_success_xy_threshold))
            & (torch.abs(z_error) < float(self.cfg.cih_success_z_threshold))
            & (upright_cos > float(self.cfg.cih_success_upright_cos))
        )

    def _collect_cylinder_visual_targets(self, stage, env_id: int):
        targets = []
        cylinder_root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Cylinder")
        if not cylinder_root.IsValid():
            return targets
        for prim in Usd.PrimRange(cylinder_root):
            if prim.IsA(UsdGeom.Gprim):
                gprim = UsdGeom.Gprim(prim)
                color_attr = gprim.GetDisplayColorAttr()
                if not color_attr:
                    color_attr = gprim.CreateDisplayColorAttr()
                targets.append((prim.GetPath().pathString, color_attr))
        return targets

    def _set_cylinder_success_visual(self, success, force: bool = False, env_ids: Sequence[int] | None = None) -> bool:
        if isinstance(success, torch.Tensor):
            success_values = success.detach().to(device="cpu", dtype=torch.bool).tolist()
        elif isinstance(success, bool):
            success_values = [success]
        else:
            success_values = [bool(value) for value in success]
        if env_ids is None:
            target_env_ids = list(range(len(success_values)))
        elif isinstance(env_ids, torch.Tensor):
            target_env_ids = [int(env_id) for env_id in env_ids.detach().to(device="cpu").tolist()]
        else:
            target_env_ids = [int(env_id) for env_id in env_ids]
        if len(success_values) == 1 and len(target_env_ids) > 1:
            success_values = success_values * len(target_env_ids)
        if len(success_values) != len(target_env_ids):
            raise ValueError("success and env_ids must have matching lengths")
        default_color = getattr(self.cfg, "cih_cylinder_default_color", (0.95, 0.72, 0.18))
        success_color = getattr(self.cfg, "cih_cylinder_success_color", (1.0, 0.0, 0.85))
        color_tuples = {
            False: (float(default_color[0]), float(default_color[1]), float(default_color[2])),
            True: (float(success_color[0]), float(success_color[1]), float(success_color[2])),
        }
        colors = {key: Gf.Vec3f(*value) for key, value in color_tuples.items()}
        changed_any = False
        for env_id, is_success in zip(target_env_ids, success_values, strict=False):
            if env_id >= len(self._cih_visual_targets):
                break
            if not force and self._cih_success_visual_state[env_id] == is_success:
                continue
            for _, color_attr in self._cih_visual_targets[env_id]:
                color_attr.Set([colors[is_success]])
            self._cih_success_visual_state[env_id] = is_success
            changed_any = True
        return changed_any
