''' zarr 2.18.7
/
 ├── data
 │   ├── obs
 │   │    ├── images: (n_total_steps, n_camera, 3, h, w) | uint8 | RGB | [0, 255]
 │   │    ├── depths: (Optional) | (n_total_steps, n_camera, 1, h, w) | float32 | in meter | >= 0
 │   │    ├── xyz_images: (Optional) | (n_total_steps, n_camera, 3, h, w) | float32 | in meter
 │   │    ├── point_clouds: (Optional) | (n_total_steps, 3, n_pc_points) | float32 | in meter
 │   │    ├── point_colors: (Optional) | (n_total_steps, 3, n_pc_points) | uint8 | [0, 255]
 │   │    ├── point_is_robot_mask: (Optional) | (n_total_steps, 1, n_pc_points) | uint8 | True if the point belongs to robot
 │   │    ├── robot_masks: (Optional) | (n_total_steps, 1, h, w) | uint8 | True if the pixel belongs to robot
 │   │    ├── ..._qpos: (Optional) e.g."robot0_gripper_qpos", "robot0_all_qpos" | (n_total_steps, n_robot_or_gripper_dof) | float32 | in radians or meters, joint orders should be described in meta
 │   │    ├── ..._qvel: (Optional) e.g."robot0_gripper_qvel", "robot0_all_qvel" | (n_total_steps, n_robot_or_gripper_dof) | float32 | in rad/s or m/s, joint orders should be described in meta
 │   │    ├── ..._pos: (Optional) e.g."robot0_eef_pos" | (n_total_steps, 3) | float32 | in meter, X_WorldEE[:3, 3]
 │   │    ├── ..._lin_vel: (Optional) e.g."robot0_eef_lin_vel" | (n_total_steps, 3) | float32 | in m/s, vector described in world frame
 │   │    ├── ..._quat: (Optional) e.g."robot0_eef_quat" | (n_total_steps, 4) | float32 | in wxyz, X_WorldEE[:3, :3] -> wxyz quat
 │   │    ├── ..._ang_vel: (Optional) e.g."robot0_eef_ang_vel" | (n_total_steps, 3) | float32 | in rad/s, axisang described in world frame
 │   │    ├── current_transforms: (n_total_steps, n_link, 4, 4) | float32 | X_WorldLinkcurrent, link orders should be described in meta
 │   │    └── current_robot_points: (n_total_steps, 3, n_points) | float32
 │   ├── actions
 │   │    ├── target_transforms: (n_total_steps, n_link, 4, 4) | float32 | X_WorldLinktarge, link orders should be described in meta
 │   │    ├── target_robot_points: (n_total_steps, 3, n_points) | float32
 │   │    ├── target_..._qpos: (Optional) e.g."target_all_qpos" | (n_total_steps, n_dof) | float32 | a PD-controller target, in radians or meters, joint orders should be described in meta
 │   │    └── original_actions: (Optional) | (n_total_steps, d_action) | float32 | action vectors in orginal simulation envs dataset
 │   └── states: (Optional) (n_total_steps, d_state) | float32 | state vectors in original simulation envs dataset
 └── meta
     ├── episode_ends: (n_episode,) | int64 | contains the end index of each episode in the data; used to seperate episodes from the original array; episode_ends[-1] should = n_total_steps
     ├── link_name_list: (n_link,) | str | an ordered list containing the link names to describe the current_transforms & target_transforms order
     ├── camera_name_list: (n_camera,) | str | an ordered list containing the camera names to describe the images order
     ├── joint_name_list: (Optional) | (n_joint,) | str | an ordered list containing the joint names to describe the qpos & qvel...
     ├── camera_meta
     │    ├── camera_name: e.g."agentview" | str | this name should be the same to one in the obs
     │    │    ├── K: (3, 3) | float32 | [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
     │    │    ├── parent_frame_name: e.g."world", "gripper0_right_gripper" | str | in eye to base setting, the camera is fixed to the world frame; in the eye in hand setting, the camera is relatively still with respect to a certain link frame, this link name should be in link_name_list
     │    │    ├── X: (4, 4) | float32 | X_ParentCamera | for camera frame, we use the COLMAP/OpenCV/viser convention: Forward: +Z, Up: -Y, Right: +X
     │    │    ├── h: () | int64 | camera height 
     │    │    └── w: () | int64 | camera width 
     │    └── ... other camera names and their attributes, if any
     ├── model_files: (Optional) | (n_episode,) | str | each model file completely defines an mujoco/... env, which may give us access to more info.
     └── ...
'''
import zarr
import numpy as np
from typing import Dict
from termcolor import cprint

class ZarrDataset:
    """
    Class to create and manage a Zarr dataset with a predefined structure.
    """
    def __init__(self, save_path: str):
        self.root = zarr.open(save_path, mode='w')
        self.define_structure()

    def define_structure(self):
        """Define the dataset structure."""
        self.root.create_group('data/obs')
        self.root.create_group('data/actions')
        self.root.create_group('meta/camera_meta')

    def save_data(self, arrays: Dict[str, np.ndarray]):
        """
        Save data to the Zarr dataset.

        Parameters:
        - arrays: A dictionary with keys as paths and values as numpy arrays to save.
        """
        for key, value in arrays.items():
            # Validate input data
            self.validate_shape_and_type(key, value)
            # Determine chunk size based on array dimensions
            chunk_size = self.get_chunk_size(value)
            # Create dataset with compression
            self.root.create_dataset(
                key,
                data=value,
                chunks=chunk_size,
                dtype=value.dtype,
                overwrite=True,
                compressor=None,  
            )

    def validate_shape_and_type(self, key: str, array: np.ndarray):
        """
        Validate the shape and type of the array to ensure compatibility.

        Parameters:
        - key: Path in the Zarr dataset.
        - array: Numpy array to validate.
        """
        if not isinstance(key, str):
            raise ValueError(f"Key must be a string, got {type(key)}.")
        if not isinstance(array, np.ndarray):
            raise ValueError(f"Array must be a numpy array, got {type(array)}.")
        if "images" in key and "xyz_images" not in key:
            assert array.ndim == 5 and array.shape[-3] == 3, f"{key} must have shape (n_steps, n_camera, 3, h, w)."
            assert array.dtype == np.uint8, f"{key} must be of type uint8."
        elif "xyz_images" in key:
            assert array.ndim == 5 and array.shape[-3] == 3, f"{key} must have shape (n_steps, n_camera, 3, h, w)."
            assert array.dtype == np.float32, f"{key} must be of type float32."
        elif "depths" in key or "robot_masks" in key:
            assert array.ndim == 5 and array.shape[-3] == 1, f"{key} must have shape (n_steps, n_camera, 1, h, w)."
            assert array.dtype in [np.float32, np.uint8, np.uint16], f"{key} must be of type float32 or uint8, current dytpe {array.dtype}"
        elif "point_clouds" in key:
            assert array.ndim == 3 and array.shape[-2] == 3, f"{key} must have shape (n_steps, 3, n_pc_points)."
            assert array.dtype == np.float32, f"{key} must be of type float32."
        elif "point_colors" in key:
            assert array.ndim == 3 and array.shape[-2] == 3, f"{key} must have shape (n_steps, 3, n_pc_points)."
            assert array.dtype == np.uint8, f"{key} must be of type uint8."
        elif "point_is_robot_mask" in key:
            assert array.ndim == 3 and array.shape[-2] == 1, f"{key} must have shape (n_steps, 1, n_pc_points)."
            assert array.dtype == np.uint8, f"{key} must be of type uint8."
        elif "qpos" in key or "qvel" in key or "pos" in key or "quat" in key or "lin_vel" in key or "ang_vel" in key or "states" in key or "original_actions" in key or "delta_actions" in key or "delta_transforms_actions" in key:
            assert array.ndim == 2, f"{key} must have shape (n_steps, d)."
            assert array.dtype == np.float32, f"{key} must be of type float32."
        elif "transforms" in key:
            assert array.ndim == 4 and array.shape[-2:] == (4, 4), f"{key} must have shape (n_steps, n_links, 4, 4)."
            assert array.dtype == np.float32, f"{key} must be of type float32."
        elif "robot_points" in key:
            assert array.ndim == 3 and array.shape[-2] == 3, f"{key} must have shape (n_steps, 3, n_points)."
            assert array.dtype == np.float32, f"{key} must be of type float32."
        elif "states" in key:
            assert array.ndim == 2, f"{key} must have shape (n_steps, d_state)."
            assert array.dtype == np.float32, f"{key} must be of type float32."
        elif "name_list" in key:
            assert array.ndim == 1, f"{key} must have shape (n_names,)."
            assert np.issubdtype(array.dtype, np.str_), f"{key} must be of type str."
        elif "episode_ends" in key:
            assert array.ndim == 1, f"{key} must have shape (n_episodes,)."
            assert array.dtype == np.int64, f"{key} must be of type int64."
        elif "model_files" in key:
            assert array.ndim == 1, f"{key} must have shape (n_episodes,)."
            assert np.issubdtype(array.dtype, np.str_), f"{key} must be of type str."
        elif "camera_meta" in key and "K" in key:
            assert array.shape == (3, 3), f"{key} must have shape (3, 3)."
            assert array.dtype == np.float32, f"{key} must be of type float32."
        elif "camera_meta" in key and "X" in key:
            assert array.shape == (4, 4), f"{key} must have shape (4, 4)."
            assert array.dtype == np.float32, f"{key} must be of type float32."
        elif "camera_meta" in key and ("/w" in key or "/h" in key):
            assert array.shape == (), f"{key} must have shape ()."
            assert array.dtype == np.int64, f"{key} must be of type int64."
        elif "camera_meta" in key and "parent_frame_name" in key:
            assert array.shape == (), f"{key} must have shape ()."
            assert np.issubdtype(array.dtype, np.str_), f"{key} must be of type str."
        elif "tag_ori" in key:
            assert array.ndim == 3 and array.shape[-1] == 60, f"{key} must have shape (n_steps, n_cam, 60)."
            assert array.dtype == np.float32, f"{key} must be of type float32."
        else:
            cprint(f"Invalid key: {key}", "red")

    def get_chunk_size(self, array: np.ndarray):
        """
        Determine appropriate chunk size based on array dimensions.

        Parameters:
        - array: Numpy array for which to calculate chunk size.

        Returns:
        - A tuple representing the chunk size.
        """
        if array.ndim >= 2:  
            shape = array.shape
            chunk_shape = [1] + list(shape[1:])
            return tuple(chunk_shape)
        else:
            return None
        
    def print_structure(self):
        """
        Print the structure of the Zarr dataset.
        """
        print(self.root.tree())

# Example usage
def main():
    save_path = "dataset.zarr"
    zarr_dataset = ZarrDataset(save_path)

    # Example data
    n_total_steps = 100
    camera_h = 64
    camera_w = 64
    n_episodes = 10
    n_links = 11
    n_gripper_joints = 2
    n_all_joints = 7+2
    example_data = {
        "data/obs/images": np.zeros((n_total_steps, 2, 3, camera_h, camera_w), dtype=np.uint8),
        "data/obs/depths": np.zeros((n_total_steps, 2, 1, camera_h, camera_w), dtype=np.float32),
        "data/obs/robot_masks": np.zeros((n_total_steps, 2, 1, camera_h, camera_w), dtype=np.uint8),
        "data/obs/robot0_gripper_qpos": np.random.rand(n_total_steps, n_gripper_joints).astype(np.float32),
        "data/obs/robot0_gripper_qvel": np.random.rand(n_total_steps, n_gripper_joints).astype(np.float32),
        "data/obs/robot0_all_qpos": np.random.rand(n_total_steps, n_all_joints).astype(np.float32),
        "data/obs/robot0_all_qvel": np.random.rand(n_total_steps, n_all_joints).astype(np.float32),
        "data/obs/robot0_eef_pos": np.random.rand(n_total_steps, 3).astype(np.float32),
        "data/obs/robot0_eef_quat": np.random.rand(n_total_steps, 4).astype(np.float32),
        "data/obs/robot0_eef_lin_vel": np.random.rand(n_total_steps, 3).astype(np.float32),
        "data/obs/robot0_eef_ang_vel": np.random.rand(n_total_steps, 3).astype(np.float32),
        "data/obs/current_transforms": np.random.rand(n_total_steps, n_links, 4, 4).astype(np.float32),
        "data/actions/target_transforms": np.random.rand(n_total_steps, n_links, 4, 4).astype(np.float32),
        "data/actions/target_all_qpos": np.random.rand(n_total_steps, n_all_joints).astype(np.float32),
        "data/actions/original_actions": np.random.rand(n_total_steps, 7).astype(np.float32),
        "data/states": np.random.rand(n_total_steps, 34).astype(np.float32),
        "meta/episode_ends": np.array([12, 20, 29, 45, 52, 61, 73, 80, 91, 100], dtype=np.int64),
        "meta/link_name_list": np.array(["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "gripper_base_link", "left_finger_link", "right_finger_link"], dtype=str),
        "meta/joint_name_list": np.array(["joint0", "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "left_finger_joint", "right_finger_joint"], dtype=str),
        "meta/camera_name_list": np.array(["agentview", "robot0_eye_in_hand"], dtype=str),
        "meta/camera_meta/agentview/K": np.random.rand(3, 3).astype(np.float32),
        "meta/camera_meta/agentview/parent_frame_name": np.array("world", dtype=str),
        "meta/camera_meta/agentview/X": np.random.rand(4, 4).astype(np.float32),
        "meta/camera_meta/agentview/h": np.array(camera_h, dtype=np.int64),
        "meta/camera_meta/agentview/w": np.array(camera_w, dtype=np.int64),
        "meta/camera_meta/robot0_eye_in_hand/K": np.random.rand(3, 3).astype(np.float32),
        "meta/camera_meta/robot0_eye_in_hand/parent_frame_name": np.array("gripper_base_link", dtype=str),
        "meta/camera_meta/robot0_eye_in_hand/X": np.random.rand(4, 4).astype(np.float32),
        "meta/camera_meta/robot0_eye_in_hand/h": np.array(camera_h, dtype=np.int64),
        "meta/camera_meta/robot0_eye_in_hand/w": np.array(camera_w, dtype=np.int64),
        "meta/model_files": np.array(["this is a model file 0", "this is a model file 1", "this is a model file 2", "this is a model file 3", "this is a model file 4", "this is a model file 5", "this is a model file 6", "this is a model file 7", "this is a model file 8", "this is a model file 9"], dtype=str),
    }

    # Save data
    zarr_dataset.save_data(example_data)

    # Print dataset structure
    zarr_dataset.print_structure()

if __name__ == "__main__":
    main()