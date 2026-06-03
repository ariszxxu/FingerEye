import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.utils import configclass

from .fingereye_env_cfg import DIR_PATH, FingerEyeLabEnvCfg, FingerEyeRandomizationCfg


def _cs_collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0)


def _normalize_vec(vec: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = math.sqrt(sum(value * value for value in vec))
    if norm <= 1.0e-12:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return tuple(value / norm for value in vec)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def opengl_camera_look_at_quat(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    up: tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[float, float, float, float]:
    """Return a (w, x, y, z) quaternion for an OpenGL camera whose -Z axis points at target."""
    forward = _normalize_vec(tuple(target[i] - eye[i] for i in range(3)))
    camera_z = tuple(-value for value in forward)
    camera_x = _normalize_vec(_cross(up, camera_z))
    camera_y = _cross(camera_z, camera_x)

    m00, m01, m02 = camera_x[0], camera_y[0], camera_z[0]
    m10, m11, m12 = camera_x[1], camera_y[1], camera_z[1]
    m20, m21, m22 = camera_x[2], camera_y[2], camera_z[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = ((0.25 * scale), (m21 - m12) / scale, (m02 - m20) / scale, (m10 - m01) / scale)
    elif m00 > m11 and m00 > m22:
        scale = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        quat = ((m21 - m12) / scale, 0.25 * scale, (m01 + m10) / scale, (m02 + m20) / scale)
    elif m11 > m22:
        scale = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        quat = ((m02 - m20) / scale, (m01 + m10) / scale, 0.25 * scale, (m12 + m21) / scale)
    else:
        scale = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        quat = ((m10 - m01) / scale, (m02 + m20) / scale, (m12 + m21) / scale, 0.25 * scale)

    quat = tuple(value / math.sqrt(sum(component * component for component in quat)) for value in quat)
    return tuple(-value for value in quat) if quat[0] < 0.0 else quat


@configclass
class FingerEyeCSLabEnvCfg(FingerEyeLabEnvCfg):
    """Coin-standing inspection variant.

    Keeps the original coin-standing task unchanged while adding a visual coin
    init-range marker and using the NH inspection third-view camera pose.
    """

    task_name = "coin_standing_cs"
    coin_x_min = 0.34
    coin_x_max = 0.46
    coin_y_min = 0.035
    coin_y_max = 0.065
    coin_z = 0.0155
    contact_coin_radius = 0.015
    contact_coin_thickness = 0.006

    default_view_eye = (0.40, 0.28, 0.13)
    default_view_target = (0.40, -0.02, 0.004)
    cs_third_view_rot_opengl = opengl_camera_look_at_quat(default_view_eye, default_view_target)

    coin_init_marker_z = 0.00012
    show_coin_init_marker = True
    enable_fingertip_tag_points = False

    randomization: FingerEyeRandomizationCfg = FingerEyeRandomizationCfg(
        enable_all=False,
        random_lighting=False,
        random_object_color=False,
        random_background=False,
        random_camera_noise=False,
    )

    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Coin",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{DIR_PATH}/assets/objects/coin/coin_isaac_sim/xarm_leap_2.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=50.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=(1, 0, 0, 0)),
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot_cfg = copy.deepcopy(self.robot_cfg)
        self.robot_cfg.spawn.collision_props = _cs_collision_props()
        self.object_cfg = copy.deepcopy(self.object_cfg)
        self.object_cfg.spawn.collision_props = _cs_collision_props()
        self.cam_third_view = copy.deepcopy(self.cam_third_view)
        self.cam_third_view.offset.pos = self.default_view_eye
        self.cs_third_view_rot_opengl = opengl_camera_look_at_quat(self.default_view_eye, self.default_view_target)
        self.cam_third_view.offset.rot = self.cs_third_view_rot_opengl
