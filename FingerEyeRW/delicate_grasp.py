import sys
import os, time
import numpy as np
import hydra
from pathlib import Path
import time
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append("/home/ps/ConTacRW/recorder_utils")
from termcolor import cprint
from leap_hand_utils.leapnode import LeapNode
from recorder_utils.recorder_tag_detector import AprilTagDetector
from recorder_utils.recorder_av_cam import AVCameraManager
from xarm.wrapper import XArmAPI
import json
import cv2
def interpolate_pose(start, end, ratio):
    return (1 - ratio) * start + ratio * end


@hydra.main(
    version_base=None,
    config_path=str(Path(__file__).parent.joinpath("configs")),
    config_name="delicate_grasp.yaml",
)
def main(config):
   

    kP = float(config.leap_kP)
    kD = float(config.leap_kD)
    leap = LeapNode(kP = kP, 
                    kD = kD, 
                    init_joint_values_radian=np.array(config.leap_init_joint_values_radian),
                    enable_hand=True)
    leap_init_pos = np.array(config.leap_init_joint_values_radian)
    leap_target_pose = np.array(config.leap_target_joint_values_radian)


    arm = XArmAPI(
        config.xarm_api
    )
    xarm_init_pos = config.xarm_init_joint_values_degree[config.task]
    if arm.connected:
                print("✅ Connect to xArm!")
                arm.set_simulation_robot(False)
                arm.clean_error()
                arm.clean_warn()
                arm.motion_enable(True)
                time.sleep(0.1)
                arm.set_mode(0) 
                time.sleep(0.1)
                arm.set_state(state=0)
                time.sleep(0.1)
                arm.set_servo_angle(
                    angle=np.array(xarm_init_pos),
                    speed=15, 
                    wait=True,
                    is_radian=False,
                )
                arm.set_mode(5) 
                arm.set_state(state=0)
                print("🤖 xArm to initial configuration!")
    else:
        print("❌ Fail to connect to xArm!")


    enabled_camera_to_port = {
        name: port for name, port in config.camera_to_port.items()
        if name in config.enabled_stereo_camera_names
    }
    enabled_camera_left_right_order = {
        name: order for name, order in config.camera_left_right_order.items()
        if name in config.enabled_stereo_camera_names
    }

    camera_manager = AVCameraManager(
        enabled_camera_to_port,
        camera_left_right_order=enabled_camera_left_right_order,
        default_options=config.default_camera_opts,
    )
    camera_manager.open_all_cameras()

    tag_detector = AprilTagDetector(config)

    warm_step = 0
    time.sleep(1.0)
    while True:
            current_camera_frames, origin_frames = camera_manager.get_frames(img_size=(config.W_resize, config.H_resize)) 
            current_center_tag_T, current_center_tag_can_use = tag_detector.get_tag_pose(origin_frames)
            warm_step += 1
            if all(current_center_tag_can_use.values()) and warm_step >= 50:
                break
            time.sleep(0.01)
    cprint("✅ Camera warm-up done.", "green")
    delta_T_z_to_save = []
    delta_I_z_to_save = []
    cprint("🏁 Starting main control loop...", "green")

    frame = 0
    max_frame = config.max_frame[config.task]
    control_dt = config.control_dt[config.task]   
    z_threshold = config.z_translation_thredshold[config.task]
    cprint(f"⚠️ Z translation threshold: {z_threshold} mm", "yellow")
    ref_T_z = None
    ref_I_z = None 
    next_control_time = time.perf_counter()
    start_time = next_control_time
    while True:
        now = time.perf_counter()
        if now < next_control_time:
            time.sleep(next_control_time - now)
            continue
        next_control_time += control_dt

        resized_frames, origin_frame = camera_manager.get_frames(
            img_size=(config.W, config.H)
        )
        current_center_tag_T, current_center_tag_can_use = tag_detector.get_tag_pose(origin_frame)

        if current_center_tag_T is None:
            print(f"[Frame {frame}] No tag detected.")
            continue
        T_matrix = current_center_tag_T["T"]
        I_matrix = current_center_tag_T["I"]
        T_z = T_matrix[2, 3] * 1000.0
        I_z = I_matrix[2, 3] * 1000.0

        if frame == 0:
            ref_T_z = T_z
            ref_I_z = I_z
            print(f"[Frame 0] Reference T_z = {ref_T_z:.2f} mm, I_z = {ref_I_z:.2f} mm")
            frame += 1
            continue

        delta_T_z = np.abs(T_z - ref_T_z)
        delta_I_z = np.abs(I_z - ref_I_z)
        delta_T_z_to_save.append(delta_T_z)
        delta_I_z_to_save.append(delta_I_z)

        if delta_T_z > z_threshold and delta_I_z > z_threshold:
            print(
                f"Stop! One of the tag Δz exceeded threshold: "
                f"ΔT_z={delta_T_z:.2f}, ΔI_z={delta_I_z:.2f}, threshold={z_threshold} mm"
            )
            break
        
        t = time.perf_counter() - start_time
        ratio = min(frame / max_frame, 1.0)
        cur_pose = interpolate_pose(leap_init_pos, leap_target_pose, ratio)
        leap.set_leap(cur_pose)

        frame += 1
    
    delta_joint = np.abs(cur_pose - leap_init_pos)

    timestamp = time.strftime("%m%d_%H%M")
    json_file_name = f"delta_z_{config.task}_{timestamp}.json"
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return obj

    data = {
        "kp": kP,
        "kd": kD,
        "speed": delta_joint / frame * control_dt,
        "delta_joint": delta_joint,
        "delta_T_z": delta_T_z_to_save,
        "delta_I_z": delta_I_z_to_save,
    }
    for key, resized_frame in resized_frames.items():
        if "tip" in key:
            cv2.imwrite(f"final_leap_pose_{config.task}_{key}_{timestamp}.png", resized_frame)
    json.dump(data, open(json_file_name, "w"), indent=4, default=to_serializable)
    cprint(f"💾 Saved data to {json_file_name}", "green")
    try:
        camera_manager.release_all()
    except:
        pass
    target_speed = np.array([0, 0, 30, 0, 0, 0])
    code = arm.vc_set_cartesian_velocity(target_speed, is_radian=None, is_tool_coord=False, duration=2) 
    cprint("🏁 Task completed. ", "green")
    time.sleep(2.0)
    arm.set_mode(0) 
    arm.set_state(state=0)
    time.sleep(1)
    arm.set_servo_angle(
        angle=np.array(xarm_init_pos),
        speed=10, 
        wait=True,
        is_radian=False,
    )
    time.sleep(1.5)
    leap_end_pose = leap.read_pos()
    for idx in range(200):
        ratio = min(idx / 200, 1.0)
        cur_pose = interpolate_pose(leap_end_pose, leap_init_pos, ratio)
        leap.set_leap(cur_pose)
    time.sleep(1.0)
    leap.safe_disconnect()

if __name__ == "__main__":
    main()

