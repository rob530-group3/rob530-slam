import cv2
import numpy as np
from feature_matcher import match_features


def initialize_vo_state(initial_img_path, initial_depth, detector):
    """
    Initialize visual odometry by detecting features in the first frame.

    Args:
        initial_img_path (str): Path to the first grayscale image.
        initial_depth (np.ndarray): Depth map corresponding to the first image.
        detector: Feature detector object (e.g., ORB, SIFT).

    Returns:
        prev_img, prev_kp, prev_des, prev_depth: Initial VO state.
    """
    img = cv2.imread(initial_img_path, cv2.IMREAD_GRAYSCALE)
    kp, des = detector.detectAndCompute(img, None)
    return img, kp, des, initial_depth


def estimate_vo_step(curr_img_path, curr_depth,
                     prev_img, prev_kp, prev_des, prev_depth,
                     K, detector, matcher, strategy,
                     R_prev, t_prev):
    """
    Estimate the current camera pose relative to the previous frame using PnP.

    Args:
        curr_img_path (str): Path to the current grayscale image.
        curr_depth (np.ndarray): Depth map for the current frame.
        prev_img, prev_kp, prev_des, prev_depth: Previous VO state.
        K (np.ndarray): Camera intrinsic matrix.
        detector, matcher, strategy: Feature matching setup.
        R_prev (np.ndarray): Previous rotation matrix.
        t_prev (np.ndarray): Previous translation vector.

    Returns:
        curr_img, curr_kp, curr_des, curr_depth: Updated VO state.
        (R_new, t_new): Estimated pose for current frame, or None if failed.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    img = cv2.imread(curr_img_path, cv2.IMREAD_GRAYSCALE)
    kp, des = detector.detectAndCompute(img, None)

    matches = match_features(matcher, prev_des, des, strategy)

    pts_3d = []
    pts_2d = []

    for m in matches:
        u_prev, v_prev = prev_kp[m.queryIdx].pt
        u_curr, v_curr = kp[m.trainIdx].pt
        u_prev, v_prev = int(u_prev), int(v_prev)

        if 0 <= u_prev < prev_depth.shape[1] and 0 <= v_prev < prev_depth.shape[0]:
            z = prev_depth[v_prev, u_prev]
            if z <= 0:
                continue
            x = (u_prev - cx) * z / fx
            y = (v_prev - cy) * z / fy
            pts_3d.append([x, y, z])
            pts_2d.append([u_curr, v_curr])

    if len(pts_3d) >= 6:
        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)

        success, rvec, tvec, _ = cv2.solvePnPRansac(
            pts_3d, pts_2d, K, None, reprojectionError=8.0, flags=cv2.SOLVEPNP_ITERATIVE)

        if success:
            R, _ = cv2.Rodrigues(rvec)
            t_new = t_prev + R_prev @ tvec
            R_new = R @ R_prev
            
            # # Added for debugging
            # # Apply extrinsic correction to account for camera mounted in reverse
            # R_correction = np.array([
            #     [-1,  0,  0],
            #     [ 0, -1,  0],
            #     [ 0,  0,  1]
            # ])
            # t_correction = np.zeros((3, 1))  # no translation shift

            # # Full transformation matrix (T = [R | t])
            # T = np.eye(4)
            # T[:3, :3] = R_new
            # T[:3, 3:] = t_new

            # # Correction matrix
            # T_correction = np.eye(4)
            # T_correction[:3, :3] = R_correction
            # T_correction[:3, 3:] = t_correction

            # # Apply correction: post-multiply (i.e., camera-to-world = T · correction)
            # T_corrected = T @ T_correction
            # R_new = T_corrected[:3, :3]
            # t_new = T_corrected[:3, 3:]
            return img, kp, des, curr_depth, (R_new, t_new)

    # VO failed for this frame
    return img, kp, des, curr_depth, None
