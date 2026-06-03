import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from .fingereye_env_cfg import DIR_PATH, FingerEyeLabEnvCfg


def _dynamic_rigid_props() -> sim_utils.RigidBodyPropertiesCfg:
    return sim_utils.RigidBodyPropertiesCfg(
        solver_position_iteration_count=16,
        solver_velocity_iteration_count=4,
    )


def _contact_props(contact_offset: float, rest_offset: float = 0.0) -> sim_utils.CollisionPropertiesCfg:
    return sim_utils.CollisionPropertiesCfg(contact_offset=contact_offset, rest_offset=rest_offset)


@configclass
class FingerEyePNLabEnvCfg(FingerEyeLabEnvCfg):
    """Independent pick-nut task.

    This task reuses the FingerEye robot, nut asset, table, cameras, and tactile
    observation assets, but it does not inherit from or spawn the PN-BD bolt task.
    """

    task_name = "pick_nut"
    success_mode = "pn"
    episode_length_s = 30.0
    decimation = 12
    pn_pick_physics_hz = 120
    action_space = 11
    observation_space = 43
    reset_on_success = False
    replay_mode = False

    assembly_scale = 0.2
    nut_xy_scale = 0.24
    nut_z_scale = assembly_scale
    nut_diameter = 0.0635 * nut_xy_scale
    nut_height = 0.029 * nut_z_scale
    nut_reset_center_z = nut_height * 0.5
    table_z = 0.0
    pn_bd_nut_mass = 0.010
    pn_pick_nut_contact_offset = 0.0002
    pn_pick_robot_contact_offset = 0.0002
    pn_pick_nut_static_friction = 10.0
    pn_pick_nut_dynamic_friction = 10.0
    pn_pick_robot_static_friction = 10.0
    pn_pick_robot_dynamic_friction = 10.0
    pn_pick_bind_robot_physics_material = True

    hp_asset_dir = DIR_PATH / "assets" / "objects" / "nut_thread"
    hp_nut_usd = str(hp_asset_dir / "m36_nut.usd")
    pn_pick_six_box_nut_usd = str(hp_asset_dir / "pn_bd_six_box_nut.usda")
    pn_pick_use_six_box_nut = False
    pn_pick_nut_scale = (1.0, 1.0, 1.0)

    contact_coin_radius = nut_diameter * 0.5
    contact_coin_thickness = nut_height
    contact_d_max = 0.020

    hp_ee_body_name = "palm_lower"
    hp_ee_action_limit = None
    hp_ik_gain = 0.7
    hp_ik_damping = 0.07
    hp_ik_max_joint_delta = 0.06
    hp_ik_pos_axis_gains = (1.0, 1.0, 1.0)
    hp_ik_rot_gain = 2.5
    hp_third_view_eye = (0.42, 0.57, 0.20)
    hp_third_view_target = (0.42, -0.06, 0.06)
    hp_third_view_rot_opengl = (0.0, 0.0, 0.6257273936, 0.7800418123)

    pn_success_height = 0.020
    pn_success_height_tolerance = 0.0
    pn_success_contact_margin = 0.004
    pn_success_xy_margin = 0.004
    pn_success_tip_height_margin = 0.001
    pn_expert_success_height = 0.030
    pn_pick_pick_success_height = 0.030
    pn_pick_success_xy_threshold = 0.010
    pn_pick_success_z_threshold = 0.009
    pn_pick_drop_height = 0.010
    pn_pick_reset_on_drop = False

    num_rerenders_on_reset = 0
    reset_stabilization_steps = 8
    reset_post_yaw_stabilization_steps = 1
    reset_return_settle_steps = 6
    nut_init_yaw_deg = 30.0
    show_pick_range_marker = False
    show_pn_pick_target_markers = False

    # Final PN reset range.
    nut_x_min = 0.360
    nut_x_max = 0.440
    nut_y_min = 0.010
    nut_y_max = 0.090
    pn_pick_light_target_xy = (0.360, 0.04814)
    pn_pick_heavy_target_xy = (0.420, 0.04814)
    pn_pick_heavy_mass_threshold = 0.030

    camera_name_list = ["third_view", "wrist_camera", "index_tip", "index_root", "thumb_tip", "thumb_root"]
    enable_fingertip_tag_points = False

    leap_init_joint_values_radian = [
        3.0708893036,
        4.4897463799,
        2.9219766703,
        3.2836476132,
        3.1415926536,
        3.1415926536,
        3.1415926536,
        3.1415926536,
        3.1415926536,
        3.1415926536,
        3.1415926536,
        3.1415926536,
        4.6317035658,
        3.1051075335,
        2.7342251806,
        3.3335546836,
    ]

    pn_expert_noise_std = (0.0, 0.0, 0.0)
    pn_expert_noise_clip = 1.5
    pn_expert_xy_stage_offset_ranges = (0.020, 0.002, 0.0)
    pn_expert_enable_finger_xy_compensation = False
    pn_expert_wrist_fine_xy_fraction = 1.0
    pn_expert_xy_down_start_step = 42.0
    pn_expert_xy_down_end_step = 73.0
    pn_expert_xy_fine_start_step = 73.0
    pn_expert_xy_fine_end_step = 90.0
    pn_expert_lift_start_step = 100
    pn_expert_eef_step_limit = 0.0042
    pn_expert_grasp_eef_z_offset = 0.0
    pn_expert_grasp_eef_z_offset_start_step = 42.0
    pn_expert_contact_z_offset = 0.002
    pn_expert_eef_z_min = 0.080
    pn_expert_eef_z_max = 0.132
    pn_expert_retry_lift_height = 0.010
    pn_expert_grasp_joint_target = (-0.020, 1.100, -0.270, 0.100, 1.470, -0.030, -0.700, 0.120)
    pn_expert_grasp_joint_step = 0.0
    pn_bd_pick_time_scale = 0.7142857
    pn_bd_pick_lift_speed_scale = 1.4
    pn_bd_pick_grip_extra_fraction = 0.0
    pn_pick_use_adaptive_pick_expert = False
    # For the pick experiment, keep the pick-stage wrist/eef XY fixed
    # until lift while still allowing the small vertical contact/lift motion.
    # This removes the initial horizontal eef trajectory-learning burden so the
    # policy can focus on fingertip deformation/tag signals and the subsequent
    # pick policy.
    pn_pick_freeze_pick_eef_until_lift = False
    pn_bd_prevent_pregrasp_reopen = True
    pn_bd_close_start_fraction = 0.25
    pn_bd_expert_diversity_enabled = True
    pn_bd_init_wrist_xy_from_pick_box = True
    pn_bd_init_wrist_z_noise_range = 0.0
    pn_bd_init_wrist_z_offset_range = (0.025, 0.042)
    pn_bd_init_wrist_ik_iters = 8
    pn_bd_goto_nut_curve_range = (0.030, 0.030, 0.012)
    pn_bd_goto_nut_curve_end_step = 30.0
    pn_expert_midpoint_offset = (0.0, 0.0, 0.000)
    pn_expert_approach_gain = 0.45
    pn_expert_approach_step_limit = 0.004
    pn_expert_approach_tol = 0.006
    pn_expert_close_recenter_gain = 0.20
    pn_expert_close_recenter_step_limit = 0.003
    pn_expert_close_step = 0.0030
    pn_expert_side_clearance = 0.0008
    pn_expert_contact_margin = 0.004
    pn_expert_overclose_steps = 40
    pn_expert_lift_height = 0.050
    pn_expert_lift_step = 0.0015
    pn_expert_leap_ik_gain = 0.35
    pn_expert_leap_ik_damping = 0.06
    pn_expert_leap_delta_limit = 0.035

    pn_pick_eef_rel_xy = (0.0358, -0.0140)
    pn_pick_transport_eef_z = 0.182
    pn_pick_place_eef_z = 0.150
    pn_pick_retreat_eef_z = 0.210
    pn_pick_eef_step_limit = 0.0030
    pn_pick_lower_eef_step_limit = 0.0015
    pn_pick_release_eef_step_limit = 0.0015
    pn_pick_waypoint_tolerance = 0.003
    pn_pick_place_xy_tolerance = 0.006
    pn_pick_arrive_hold_steps = 4
    pn_pick_sensing_hold_steps = 40
    pn_pick_release_steps = 16
    pn_pick_release_open_fraction = 1.0
    pn_pick_retreat_steps = 20
    pn_pick_return_qpos_steps = 20
    pn_pick_return_qpos_tolerance = 0.015
    pn_pick_kinematic_carry_after_pick = False
    pn_pick_kinematic_carry_height = 0.020
    pn_pick_kinematic_place_steps = 8

    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Nut",
        spawn=sim_utils.UsdFileCfg(
            usd_path=hp_nut_usd,
            scale=(nut_xy_scale, nut_xy_scale, nut_z_scale),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            rigid_props=_dynamic_rigid_props(),
            mass_props=sim_utils.MassPropertiesCfg(mass=pn_bd_nut_mass),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=(1.0, 0.0, 0.0, 0.0)),
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=10, replicate_physics=True)

    def __post_init__(self):
        super().__post_init__()
        self.sim = copy.deepcopy(self.sim)
        self.sim.dt = 1 / float(self.pn_pick_physics_hz)
        self.sim.render_interval = self.decimation
        self.robot_cfg = copy.deepcopy(self.robot_cfg)
        self.robot_cfg.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        )
        self.robot_cfg.spawn.collision_props = _contact_props(float(self.pn_pick_robot_contact_offset))
        self.robot_cfg.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self.pn_pick_robot_static_friction),
            dynamic_friction=float(self.pn_pick_robot_dynamic_friction),
        )
        self.cam_third_view = copy.deepcopy(self.cam_third_view)
        self.cam_third_view.offset.pos = self.hp_third_view_eye
        self.cam_third_view.offset.rot = self.hp_third_view_rot_opengl
        self.object_cfg = copy.deepcopy(self.object_cfg)
        if bool(getattr(self, "pn_pick_use_six_box_nut", False)):
            self.object_cfg.spawn.usd_path = self.pn_pick_six_box_nut_usd
            self.object_cfg.spawn.scale = tuple(self.pn_pick_nut_scale)
        else:
            self.object_cfg.spawn.usd_path = self.hp_nut_usd
            self.object_cfg.spawn.scale = (self.nut_xy_scale, self.nut_xy_scale, self.nut_z_scale)
        self.object_cfg.spawn.rigid_props = _dynamic_rigid_props()
        self.object_cfg.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=float(self.pn_bd_nut_mass))
        self.object_cfg.spawn.collision_props = _contact_props(float(self.pn_pick_nut_contact_offset))
        self.object_cfg.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=float(self.pn_pick_nut_static_friction),
            dynamic_friction=float(self.pn_pick_nut_dynamic_friction),
        )
