import copy
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass

from .fingereye_env_cfg import FingerEyeLabEnvCfg, FingerEyeRandomizationCfg


FILE_PATH = Path(__file__).resolve()
DIR_PATH = FILE_PATH.parent.parent.parent.parent.parent.parent.parent


def _dynamic_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=5.0,
        linear_damping=0.0,
        angular_damping=0.02,
        enable_gyroscopic_forces=True,
        solver_position_iteration_count=64,
        solver_velocity_iteration_count=4,
    )


def _kinematic_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        kinematic_enabled=True,
        disable_gravity=True,
        max_depenetration_velocity=5.0,
        solver_position_iteration_count=64,
        solver_velocity_iteration_count=4,
    )


def _cih_collision_props() -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(contact_offset=0.0002, rest_offset=0.0)


def _cih_physics_material() -> RigidBodyMaterialCfg:
    return RigidBodyMaterialCfg(static_friction=0.3, dynamic_friction=0.25)


@configclass
class FingerEyeCIHLabEnvCfg(FingerEyeLabEnvCfg):
    """Final cylinder-in-hole release task."""

    task_name = "cylinder_in_hole"
    episode_length_s = 20.0
    action_space = 8
    observation_space = 43
    reset_on_success = False

    table_z = 0.0
    cylinder_radius = 0.005
    cylinder_height = 0.006
    cylinder_mass = 0.0028
    hole_radius = 0.0056
    hole_depth = 0.003
    plate_thickness = hole_depth
    plate_outer_radius = 0.500
    cylinder_yaw_min = -3.141592653589793
    cylinder_yaw_max = 3.141592653589793

    cih_asset_dir = DIR_PATH / "assets" / "objects" / "cylinder_hole"
    cih_cylinder_usd = str(cih_asset_dir / "cih_cylinder.usda")
    cih_plate_usd = str(cih_asset_dir / "cih_round_hole_plate.usda")

    marker_nut_left_x_min = 0.341405451
    marker_nut_left_x_max = 0.386107594
    marker_nut_left_y_min = 0.045171823
    marker_nut_left_y_max = 0.054540444
    marker_nut_right_x_min = 0.414250821
    marker_nut_right_x_max = 0.459435403
    marker_nut_right_y_min = 0.045089573
    marker_nut_right_y_max = 0.054925673
    marker_hole_x_min = 0.390060604
    marker_hole_x_max = 0.409446687
    marker_hole_y_min = 0.045161188
    marker_hole_y_max = 0.054980125

    cylinder_left_x_min = marker_nut_left_x_min
    cylinder_left_x_max = marker_nut_left_x_max
    cylinder_left_y_min = marker_nut_left_y_min
    cylinder_left_y_max = marker_nut_left_y_max
    cylinder_right_x_min = marker_nut_right_x_min
    cylinder_right_x_max = marker_nut_right_x_max
    cylinder_right_y_min = marker_nut_right_y_min
    cylinder_right_y_max = marker_nut_right_y_max
    cylinder_y_min = min(cylinder_left_y_min, cylinder_right_y_min)
    cylinder_y_max = max(cylinder_left_y_max, cylinder_right_y_max)
    hole_x_min = marker_hole_x_min
    hole_x_max = marker_hole_x_max
    hole_y_min = marker_hole_y_min
    hole_y_max = marker_hole_y_max

    cih_success_xy_threshold = 0.0008
    cih_success_z_threshold = 0.00029
    cih_success_upright_cos = 0.9961947

    default_view_eye = (0.40, 0.16, 0.16)
    default_view_target = (0.40, 0.05, 0.004)
    cih_third_view_rot_opengl = (0.0, 0.0, 0.3022762529, 0.9532203664)

    show_cih_hole_ring = True
    cih_hole_ring_color = (0.0, 0.18, 1.0)
    cih_hole_ring_opacity = 0.95
    cih_hole_ring_segments = 96
    cih_hole_ring_z_offset = 0.00008
    show_cih_hole_bottom = True
    cih_hole_bottom_color = (1.0, 1.0, 1.0)
    cih_hole_bottom_opacity = 1.0
    cih_hole_bottom_radius_scale = 0.92
    cih_hole_bottom_z_offset = 0.00005

    cih_success_visual_enabled = True
    cih_success_visual_replay_only = False
    cih_cylinder_default_color = (0.95, 0.72, 0.18)
    cih_cylinder_success_color = (1.0, 0.0, 0.85)

    enable_fingertip_tag_points = False
    contact_object_radius = cylinder_radius
    contact_object_thickness = cylinder_height

    randomization: FingerEyeRandomizationCfg = FingerEyeRandomizationCfg(
        enable_all=False,
        random_lighting=False,
        random_object_color=False,
        random_background=False,
        random_camera_noise=False,
        visible_in_primary_ray=False,
    )

    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cylinder",
        spawn=sim_utils.UsdFileCfg(
            usd_path=cih_cylinder_usd,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            rigid_props=_dynamic_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=cylinder_mass),
            collision_props=_cih_collision_props(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=(1.0, 0.0, 0.0, 0.0)),
    )

    plate_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/HolePlate",
        spawn=sim_utils.UsdFileCfg(
            usd_path=cih_plate_usd,
            rigid_props=_kinematic_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=_cih_collision_props(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=(1.0, 0.0, 0.0, 0.0)),
    )

    def __post_init__(self):
        super().__post_init__()
        self.robot_cfg = copy.deepcopy(self.robot_cfg)
        self.robot_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
        self.cam_third_view = copy.deepcopy(self.cam_third_view)
        self.cam_third_view.offset.pos = self.default_view_eye
        self.cam_third_view.offset.rot = self.cih_third_view_rot_opengl
