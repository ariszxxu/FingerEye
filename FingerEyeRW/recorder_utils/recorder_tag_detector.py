from copy import deepcopy

import cv2
import numpy as np
from termcolor import cprint
from omegaconf import OmegaConf

from pupil_apriltags import Detector

name_to_big_eye_index = {
    "I": "I-tip",
    "T": "T-tip",
    "M": "M-tip",
}

class AprilTagDetector:
    def __init__(self, cfg: OmegaConf):
        self.cfg = cfg
        self.tag_detector = Detector(families=self.cfg.Detector.families, quad_sigma=np.float32(self.cfg.Detector.quad_sigma), quad_decimate=np.float32(self.cfg.Detector.quad_decimate))
        self.cam_k_dict = {key: np.array(value['cam_k'], dtype=np.float32) 
                    for key, value in self.cfg["april_tag_config"].items()
                    if key in self.cfg.enabled_stereo_camera_names
                    }  # {"I": np.array, "T": np.array, ...}
        self.big_eye_index_dict = {key: value['big_eye'] 
                        for key, value in self.cfg["april_tag_config"].items()
                        if key in self.cfg.enabled_stereo_camera_names
                        } # {"I": "I-tip", "T": "T-tip", ...}
        self.dist_dict = {key: np.array(value['dist'], dtype=np.float32) 
                    for key, value in self.cfg["april_tag_config"].items()
                    if key in self.cfg.enabled_stereo_camera_names
                    }  # {"I": np.array, "T": np.array, ...}

        self.last_single_tag_T = {
            cam: {tag_id: np.eye(4) for tag_id in self.cfg.april_tag_config.transfer_to_center.tag_id.keys()}
            for cam in self.cfg.enabled_stereo_camera_names
        }  # I | 01234 | -> 4x4 
        self.last_center_tag_T = {key: np.eye(4) for key in self.big_eye_index_dict.keys()}  # key: I | 4x4
        self.map1_dict, self.map2_dict = {}, {}
        for key in self.cfg.enabled_stereo_camera_names:
            map1, map2 = cv2.initUndistortRectifyMap(
                self.cam_k_dict[key], self.dist_dict[key], None, self.cam_k_dict[key],
                (self.cfg.W, self.cfg.H), cv2.CV_32FC1
                )
            self.map1_dict[key], self.map2_dict[key] = map1, map2
        self.gt_positions_mm = self.get_gt_positions_mm_from_config(self.cfg)
    
    def get_tag_pose(self,img_dict):
        """
        img_dict: e.g, {
            'I-root': np.ndarray HxWx3, 480x640x3,  | BGR
            'I-tip': ...
            'T-tip', 
            'T-root'
        }
        
        """
        self.current_raw_single_tag_T = {
            cam: {tag_id: np.eye(4) for tag_id in self.cfg.april_tag_config.transfer_to_center.tag_id.keys()}
            for cam in self.cfg.enabled_stereo_camera_names
        }  # I | 01234 | -> 4x4 
        self.current_raw_single_tag_can_use = {
            cam: {tag_id: False for tag_id in self.cfg.april_tag_config.transfer_to_center.tag_id.keys()}
            for cam in self.cfg.enabled_stereo_camera_names
        }
        self.current_center_tag_T = {key: np.eye(4) for key in self.big_eye_index_dict.keys()}  # key: I | 4x4
        self.current_center_tag_can_use = {key: False for key in self.big_eye_index_dict.keys()}  # key: I | 4x4

        tag_img_dict = {}
        # tag_calculated_img_dict = {}
        for key in self.cfg.enabled_stereo_camera_names:
            # find the corresponding big eye image and k 
            big_eye_index = name_to_big_eye_index[key]
            undistorted = deepcopy(img_dict[big_eye_index])
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))
            gray = clahe.apply(gray)
            gray_rbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            cam_k = self.cam_k_dict[key]
            camera_params = self.get_camera_params_from_cam_k(cam_k)
            tag_size = self.cfg.Detector.tag_size
            tags = self.tag_detector.detect(gray, camera_params=camera_params, tag_size=tag_size,estimate_tag_pose=False) # one is  0.0196 seconds, another is 0.0213 seconds
            if len(tags) == 0 and not np.allclose(self.last_center_tag_T[key], np.eye(4), atol=1e-8):
                cprint(f"Camera {key}: No tags detected, using last known center tag pose.", "red")
                self.current_center_tag_T[key] = deepcopy(self.last_center_tag_T[key])     
                self.current_center_tag_can_use[key] = False
                tag_img_dict[key] = np.concatenate((gray_rbg, undistorted, undistorted), axis=1)
            else: 
                self.current_center_tag_can_use[key] = True
                self.single_center_tag_pose(tags, key) 
                if self.current_center_tag_can_use[key] == False:
                    self.current_center_tag_T[key] = deepcopy(self.last_center_tag_T[key])
                else:
                    self.last_center_tag_T[key] = deepcopy(self.current_center_tag_T[key])   
            if not np.allclose(self.last_center_tag_T[key], np.eye(4), atol=1e-8):
                cprint(f"Camera {key}: Outlier detected in center tag pose, using last known center tag pose.", "red")
                self.current_center_tag_T[key] = deepcopy(self.last_center_tag_T[key])
                self.current_center_tag_can_use[key] = False
        return self.current_center_tag_T, self.current_center_tag_can_use
    
    def get_camera_params_from_cam_k(self, cam_k):
        camera_params = (
            cam_k[0, 0],  # fx
            cam_k[1, 1],  # fy
            cam_k[0, 2],  # cx
            cam_k[1, 2],  # cy
        )
        return camera_params

    def single_center_tag_pose(self, tags, key):

        for tag in tags:
            if tag.tag_id not in self.cfg.april_tag_config.transfer_to_center.tag_id.keys():
                continue
            self.current_raw_single_tag_can_use[key][tag.tag_id] = True

        id2det = {tag.tag_id: tag for tag in tags}
        valid_tags =[]
        for tag_id, can_use in self.current_raw_single_tag_can_use[key].items():
            if can_use:
                if tag_id in id2det:
                    valid_tags.append(id2det[tag_id])
                else:
                    print(f"[WARN] tag_id {tag_id} can be detected but cannot be used!")
        valid_tags = self.convert_pupil_detection(valid_tags)
        current_center_tag_T = np.eye(4)
        if len(valid_tags) == 0:
            self.current_center_tag_can_use[key] = False
            return
        current_center_tag_R, current_center_tag_t, self.current_raw_single_tag_T[key] = self.refine_board_pose_apriltag(valid_tags, self.cam_k_dict[key], self.dist_dict[key],
                                        tag_size_m=self.cfg.Detector.tag_size,
                                        target_id=self.cfg.april_tag_config.center_tag_id)
        current_center_tag_T[:3, :3] = current_center_tag_R
        current_center_tag_T[:3, 3] = current_center_tag_t.reshape(3,)
        self.current_center_tag_T[key] = deepcopy(current_center_tag_T)

    def refine_board_pose_apriltag(self, detections, K, D, tag_size_m=0.002, target_id=0):
        """
        Bundle refine pose of the AprilTag board using pupil_apriltags output.
        
        Inputs:
            detections : list of detection dicts from pupil_apriltags
            K, D       : intrinsics (3×3), distortion (5×)
            tag_size_m : tag size in meters
            target_id  : the center tag id (your reference)
        
        Output:
            R_cam_target, t_cam_target  : pose of target tag in camera frame
            T_cam_tag                   : dict {tid: (R, t)}
        """

        # -------- 1. Corner 3D coordinates in tag frame --------
        half = tag_size_m / 2.0
        obj_corners_tag = np.array([
        [-half,  -half, 0],
        [ half,  -half, 0],
        [ half, half, 0],
        [-half, half, 0],
        ], dtype=np.float32)
    
        # -------- 3. Collect all correspondences for bundle PnP --------
        all_obj_pts = []
        all_img_pts = []

        for det in detections:
            tid = det["id"]
            if tid not in self.gt_positions_mm:
                continue

            img_corners = det["lb-rb-rt-lt"].astype(np.float32)  # shape = (4,2)
            
            # tag→target transform
            t_tag_target = self.gt_positions_mm[tid] * 0.001  # mm→m
            R_tag_target = [[1,0,0],[0,-1,0],[0,0,-1]]

            # Compute 3D corners in the target-tag frame
            obj_pts = (R_tag_target @ obj_corners_tag.T).T + t_tag_target
            all_obj_pts.append(obj_pts)
            all_img_pts.append(img_corners)
        # Stack into Nx3 and Nx2
        all_obj_pts = np.vstack(all_obj_pts).astype(np.float32)
        all_img_pts = np.vstack(all_img_pts).astype(np.float32)

        # -------- 4. Solve PnP for T_cam_target --------
        success, rvec, tvec = cv2.solvePnP(
            all_obj_pts, all_img_pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE
        )
        rvec, tvec = cv2.solvePnPRefineLM(
            all_obj_pts, all_img_pts, K, D, rvec, tvec
        )
        R_cam_target, _ = cv2.Rodrigues(rvec)
        t_cam_target = tvec.reshape(3, 1)
        # -------- 5. Compute T_cam_tag for all tags --------
        T_cam_tag = {}
        for tid, t_mm in self.gt_positions_mm.items():
            t_tag_target = (t_mm * 0.001).reshape(3,1)
            R_tag_target = np.eye(3)
            R = R_cam_target @ R_tag_target
            t = R_cam_target @ t_tag_target + t_cam_target
            T_cam_tag[tid] = (R, t)
        

        return R_cam_target, t_cam_target, T_cam_tag
    
    def convert_pupil_detection(self, detections):
        converted_detections = []
        
        for det in detections:
            det_dict = {
                'center': np.array(det.center, dtype=np.float32),
                'id': det.tag_id,
                'lb-rb-rt-lt': np.array(det.corners, dtype=np.float32)
            }
            converted_detections.append(det_dict)
        
        return converted_detections
    
    def get_gt_positions_mm_from_config(self, cfg):
        gt_positions_mm = {}
        tag_config = cfg.april_tag_config.transfer_to_center.tag_id
        
        for tag_id, config in tag_config.items():
            p_array = np.array(config.p).flatten()
            gt_positions_mm[tag_id] = p_array * 1000.0
        
        return gt_positions_mm

    def compute_reprojection_error(
        detections, T_cam_tag, gt_positions_mm, K, D, image,
        tag_size_m=0.002
    ):
        vis = image.copy()

        # -------------------------------
        # 1. Canonical AprilTag corner order: TL, TR, BR, BL
        # -------------------------------
        half = tag_size_m / 2.0
        obj_corners_tag = np.array([
        [-half,  -half, 0],
        [ half,  -half, 0],
        [ half, half, 0],
        [-half, half, 0],
        ], dtype=np.float32)

        per_tag_errors = {}
        all_errors = []

        # -------------------------------
        # 2. Iterate over all detections
        # -------------------------------
        for det in detections:
            tid = det["id"]
            if tid not in gt_positions_mm:
                continue
            if tid not in T_cam_tag:
                continue

            R_cam_tag, t_cam_tag = T_cam_tag[tid]

            # 4 true corners in the detection (lb-rb-rt-lt)
            # NOTE: your detector's order is LB, RB, RT, LT
            img_det = det["lb-rb-rt-lt"].astype(np.float32)

            # -------------------------------
            # 3. Reproject 3D corners
            # -------------------------------
            rvec, _ = cv2.Rodrigues(R_cam_tag)
            img_proj, _ = cv2.projectPoints(
                obj_corners_tag, rvec, t_cam_tag, K, D
            )
            img_proj = img_proj.reshape(-1, 2)

            # -------------------------------
            # 4. Compute error
            # -------------------------------
            corner_errors = np.linalg.norm(img_proj - img_det, axis=1)
            tag_mean_error = float(np.mean(corner_errors))

            per_tag_errors[tid] = {
                "corner_errors": corner_errors.tolist(),
                "mean": tag_mean_error
            }
            all_errors.extend(corner_errors.tolist())

            # -------------------------------
            # 5. Visualization
            # -------------------------------
            for (u_det, v_det), (u_proj, v_proj) in zip(img_det, img_proj):
                # detected corner: blue
                cv2.circle(vis, (int(u_det), int(v_det)), 4, (255, 0, 0), -1)

                # reprojected corner: red
                cv2.circle(vis, (int(u_proj), int(v_proj)), 4, (0, 0, 255), -1)

                # line between them: cyan
                cv2.line(
                    vis,
                    (int(u_det), int(v_det)),
                    (int(u_proj), int(v_proj)),
                    (255, 255, 0),
                    1
                )

            # Put tag id label
            cx, cy = det["center"]
            cv2.putText(vis, f"id={tid}",
                (int(cx), int(cy)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0,255,0), 1, cv2.LINE_AA
            )

        # -------------------------------
        # 6. Total RMS error
        # -------------------------------
        all_errors = np.array(all_errors)
        # total_rms = float(np.sqrt(np.mean(all_errors**2))) if len(all_errors)>0 else None
        total_rms = np.mean(all_errors)

        return per_tag_errors, total_rms, vis