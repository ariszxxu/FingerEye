from collections.abc import Sequence

import numpy as np
import torch
from pxr import Gf, UsdGeom

from isaaclab.sim.utils import get_current_stage
from isaaclab.utils.math import quat_apply, sample_uniform

from .env_tools import quat_to_M6
from .fingereye_cs_env_cfg import FingerEyeCSLabEnvCfg
from .fingereye_env import FingerEyeLabEnv


class FingerEyeCSLabEnv(FingerEyeLabEnv):
    """Coin-standing variant with debug visualization only."""

    cfg: FingerEyeCSLabEnvCfg

    def _setup_scene(self):
        super()._setup_scene()
        if not getattr(self.cfg, "show_coin_init_marker", True):
            return
        stage = get_current_stage()
        for env_id in range(self.num_envs):
            self._setup_coin_init_range_marker(stage, env_id)

    def _setup_coin_init_range_marker(self, stage, env_id: int):
        env_path = f"/World/envs/env_{env_id}"
        self._make_range_box(
            stage,
            f"{env_path}/Debug/coin_init_range",
            float(self.cfg.coin_x_min),
            float(self.cfg.coin_x_max),
            float(self.cfg.coin_y_min),
            float(self.cfg.coin_y_max),
            float(getattr(self.cfg, "coin_init_marker_z", 0.00012)),
            Gf.Vec3f(1.0, 0.0, 0.0),
        )

    def _make_range_box(self, stage, prim_path: str, x_min: float, x_max: float, y_min: float, y_max: float, z: float, color):
        thickness = 0.001
        height = 0.0002
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
            gprim.CreateDisplayOpacityAttr().Set([0.9])

    def _get_observations(self) -> dict:
        obs = super()._get_observations()
        obs["coin_pose"] = torch.cat((self.object_pos, self.object_rot), dim=-1)
        obs["coin_z_axis"] = quat_to_M6(self.object_rot)
        obs["pos_of_coin"] = self.object_pos
        return obs

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()
        normal_z_abs = torch.abs(self.coin_normal_w[:, 2])
        angle_tolerance = 10 / 180 * np.pi
        sin_angle_tolerance = np.sin(angle_tolerance)

        fingertip_pos = self.hand.data.body_pos_w[:, self.fingertip_indices]
        first_fingertip_pos = fingertip_pos[:, 0, :]
        second_fingertip_pos = fingertip_pos[:, 1, :]
        dist_between_fingertips = torch.norm(first_fingertip_pos - second_fingertip_pos, dim=-1)

        return ((normal_z_abs < sin_angle_tolerance) & (dist_between_fingertips > 0.12)).float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super(FingerEyeLabEnv, self)._reset_idx(env_ids)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        random_env_ids = env_ids[1:]

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
            self._randomize_env_skybox(env_ids)
            self._randomize_env_floor_color(
                random_env_ids,
                base_gray=self.rand_cfg.random_floor_base_gray,
                delta=self.rand_cfg.random_floor_delta,
            )

        object_state = self.object.data.default_root_state.clone()[env_ids]
        object_state[:, 0] = sample_uniform(self.cfg.coin_x_min, self.cfg.coin_x_max, (len(env_ids),), device=self.device)
        object_state[:, 1] = sample_uniform(self.cfg.coin_y_min, self.cfg.coin_y_max, (len(env_ids),), device=self.device)
        object_state[:, 2] = self.cfg.coin_z
        object_state[:, 0:3] += self.scene.env_origins[env_ids]
        object_state[:, 7:] = 0.0
        self.object.write_root_pose_to_sim(object_state[:, :7], env_ids)
        self.object.write_root_velocity_to_sim(object_state[:, 7:], env_ids)

        self._reset_hand(env_ids)
        self.successes[env_ids] = 0
        self._compute_intermediate_values(env_ids)
        self.sim.step()

    def _compute_intermediate_values(self, env_ids: Sequence[int] | None = None):
        super()._compute_intermediate_values(env_ids)
        self.coin_normal_w = quat_apply(self.object_rot, self.local_normal_vector)
