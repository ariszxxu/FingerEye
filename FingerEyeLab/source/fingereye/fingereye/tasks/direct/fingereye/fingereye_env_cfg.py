import math
import gymnasium as gym
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets import RigidObjectCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.allegro import ALLEGRO_HAND_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg, RenderCfg
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import TiledCameraCfg

from pathlib import Path
FILE_PATH = Path(__file__).resolve()
DIR_PATH = FILE_PATH.parent.parent.parent.parent.parent.parent.parent  

@configclass
class FingerEyeRandomizationCfg:
    enable_all: bool = True
    visible_in_primary_ray: bool = False
        
    random_lighting: bool = True          
    random_coin_color: bool = True        
    random_background: bool = True 
    random_camera_noise: bool = True

# Default environment configuration for FingerEye in Isaac Lab.
@configclass
class FingerEyeLabEnvCfg(DirectRLEnvCfg):
    # control frequency 
    control_dt = 0.1  # seconds  
    n_env_record = 16
    rollout_record_interval = 1
    replay_mode = False
    # env
    decimation = 6
    episode_length_s = 20.0
    # Action space: 8 joints, range usually -1.0 to 1.0 (scaled in env)
    action_space = 8
    
    # Observation space: 37 dims
    state_space = 0
    observation_space = 43

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        physx=PhysxCfg(
            enable_stabilization=True,  # NOTE: IMPORTANT
        ),
        render=RenderCfg(
            antialiasing_mode="Off", # DLAA is sharper than FXAA, still less ghosty than TAA
        ),
    )

    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{DIR_PATH}/assets/xarm7_leap_right/xarm7_leap_right_usd/xarm7_leap_right.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.005),), # realworld setting
        actuators={
            "robot_body": ImplicitActuatorCfg(
                joint_names_expr=["joint.*", "leap_.*"],
                effort_limit=None,
                stiffness={
                    "joint.*": 6000.0,
                    "leap_.*": 400.0,
                },
                damping={
                    ".*": 80.0,
                },
            ),
            "holders": ImplicitActuatorCfg(
                joint_names_expr=[".*_holder_.*"],
                effort_limit=None,
                stiffness = {
                    ".*_holder_px": 70.0,     # Fx
                    ".*_holder_py": 29.17,    # Fy
                    ".*_holder_pz": 100.0,    # Fz
                    ".*_holder_rx": 0.50,     # Tx
                    ".*_holder_ry": 0.20,     # Ty
                    ".*_holder_rz": 13.33,    # Tz
                },
                damping = {
                    ".*_holder_px": 2.37,     # Fx
                    ".*_holder_py": 1.53,     # Fy
                    ".*_holder_pz": 2.83,     # Fz
                    ".*_holder_rx": 0.010,    # Tx
                    ".*_holder_ry": 0.0063,   # Ty
                    ".*_holder_rz": 0.052,    # Tz
                }
            ),
        },
    )

    object_cfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Coin",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{DIR_PATH}/assets/objects/coin/coin_isaac_sim/xarm_leap_2.usda",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(density=50.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 0.0)  # RGB for Yellow
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(rot=(1, 0, 0, 0)),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=16, env_spacing=10, replicate_physics=True,
    )

    # init values 
    leap_init_joint_values_radian = [3.1492626667022705, 3.922388792037964, 2.957515001296997, 3.1692042350769043, 
                                    3.1415927410125732, 3.1415927410125732, 3.1415927410125732, 3.1415927410125732, 
                                    3.1415927410125732, 3.1415927410125732, 3.1415927410125732, 3.1415927410125732, 
                                    4.663301467895508, 3.15079665184021, 2.165980815887451, 3.184544086456299]

    xarm_init_joint_values_degree = [0.008938, 3.018456, 0.007219, 25.136461, -0.05191, 20.339543, -0.001547]
    control_joint_names = ['leap_0','leap_1','leap_2','leap_3','leap_12','leap_13','leap_14','leap_15']
    all_actuated_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7',
                                'leap_0','leap_1','leap_2','leap_3', # index
                                'leap_4','leap_5','leap_6','leap_7', # middle
                                'leap_8','leap_9','leap_10','leap_11', # ring
                                'leap_12','leap_13','leap_14','leap_15'] # thumb

    coin_x_min = 0.34
    coin_x_max = 0.46 
    coin_y_min = 0.04
    coin_y_max = 0.06
    coin_z = 0.0155
    # Randomize 
    randomization: FingerEyeRandomizationCfg = FingerEyeRandomizationCfg(
        enable_all=False,
        random_lighting=False,
        random_coin_color=False,
        random_background=False,
        random_camera_noise=False,
    )
    # camera 
    enable_cameras = True
    # -----------------------------------------------------------------------
    # Global camera resolution and selection
    # -----------------------------------------------------------------------
    img_w = 256
    img_h = 192

    # Only cameras in this list will be initialized and rendered.
    camera_name_list = ["third_view", "wrist_camera", "index_root", "index_tip", "thumb_root", "thumb_tip"]

    # -----------------------------------------------------------------------
    # Camera Definitions
    # -----------------------------------------------------------------------
    
    cam_third_view: TiledCameraCfg = TiledCameraCfg(
        # Attach this camera to the environment root, not the robot body.
        prim_path="/World/envs/env_.*/Robot/xarm7/ThirdViewCamera",
        
        # This camera is not defined in USD, so it must be spawned here.
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, 
            focus_distance=400.0, 
            horizontal_aperture=20.955,
            clipping_range=(0.1, 10.0),
        ),

        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.42, 0.5, 0.1),
            rot=(0.0, 0.0, 0.70711, 0.70711),
            convention="opengl",
        ),
        data_types=["rgb"],
        width=img_w, 
        height=img_h,
    )

    # 1. Wrist
    cam_wrist: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/xarm7/palm_lower/wrist_camera",
        spawn=None,
        data_types=["rgb"],
        width=img_w, 
        height=img_h,
        update_latest_camera_pose=True,
    )

    # 2. Index Finger (Tip & Root)
    cam_index_tip: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/xarm7/fingertip_holder/I_tip",
        spawn=None,
        data_types=["rgb"],
        width=img_w, 
        height=img_h,
        update_latest_camera_pose=True,
    )

    cam_index_root: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/xarm7/fingertip_holder/I_root",
        spawn=None,
        data_types=["rgb"],
        width=img_w, 
        height=img_h,
        update_latest_camera_pose=True,
    )

    # 3. Thumb (Tip & Root)
    cam_thumb_tip: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/xarm7/thumb_fingertip/T_tip",
        spawn=None,
        data_types=["rgb"],
        width=img_w, 
        height=img_h,
        update_latest_camera_pose=True,
    )

    cam_thumb_root: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/xarm7/thumb_fingertip/T_root",
        spawn=None,
        data_types=["rgb"],
        width=img_w, 
        height=img_h,
        update_latest_camera_pose=True,
    )

    # tags: [] for no pose, len=2 for 2 fingers. 
    enabled_tag_joints_prefix = ["fingertip_holder", # index
                                 "thumb_holder"]  # thumb
    # fingertip_2 | middle ; fingertip_3 | ring 


@configclass
class FingerEyeTeleopLabEnvCfg(FingerEyeLabEnvCfg):
    episode_length_s = 60.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=10, replicate_physics=True, # clone_in_fabric=True
    )
    camera_name_list = ["third_view",]

@configclass
class FingerEyeReplayLabEnvCfg(FingerEyeLabEnvCfg):
    replay_mode = True

@configclass
class FingerEyeReplayRandomLabEnvCfg(FingerEyeLabEnvCfg):
    replay_mode = True
    randomization: FingerEyeRandomizationCfg = FingerEyeRandomizationCfg(
        enable_all  = False,
        random_lighting= True,
        random_coin_color= True,   
        random_background= True,
        random_camera_noise= True,
        visible_in_primary_ray = False, # dark sky
    )
