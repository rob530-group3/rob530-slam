import cv2
import numpy as np
from feature_matcher import match_features
from tqdm import tqdm

def estimate_trajectory(left_imgs, depth_maps, K, detector, matcher, strategy):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    R_f = np.eye(3)
    t_f = np.zeros((3, 1))
    trajectory = []

    prev_img = None
    prev_kp = None
    prev_des = None
    prev_depth = None

    print("---- Estimating trajectory ----")
    for i in range(tqdm(len(left_imgs))):
        img = cv2.imread(left_imgs[i], cv2.IMREAD_GRAYSCALE)
        depth = depth_maps[i]

        kp, des = detector.detectAndCompute(img, None)

        if prev_img is not None and prev_des is not None:
            matches = match_features(matcher, prev_des, des, strategy)

            pts_3d = []
            pts_2d = []

            for m in matches:
                u_prev, v_prev = prev_kp[m.queryIdx].pt
                u_curr, v_curr = kp[m.trainIdx].pt
                u_prev, v_prev = int(u_prev), int(v_prev)

                if 0 <= u_prev < prev_depth.shape[1] and 0 <= v_prev < prev_depth.shape[0]:
                    z = prev_depth[v_prev, u_prev]

                    if 0.1 < z < 80:  # Filter bad depth
                        x = (u_prev - cx) * z / fx
                        y = (v_prev - cy) * z / fy
                        pts_3d.append([x, y, z])
                        pts_2d.append([u_curr, v_curr])
            
            # depth_vals = [pt[2] for pt in pts_3d if pt[2] > 0]
            # if len(depth_vals) > 0:
            #     print(f"[DEBUG] PnP Depth | mean: {np.mean(depth_vals):.2f}, min: {np.min(depth_vals):.2f}, max: {np.max(depth_vals):.2f}")
            # else:
            #     print(f"[WARNING] No valid depth points for frame {i}")

            if len(pts_3d) >= 6:
                pts_3d = np.array(pts_3d, dtype=np.float32)
                pts_2d = np.array(pts_2d, dtype=np.float32)
                # print(f"[DEBUG] PnP input | 3D: {pts_3d.shape}, 2D: {pts_2d.shape}")    # debug

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d, pts_2d, K, None, reprojectionError=8.0, flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    R, _ = cv2.Rodrigues(rvec)

                    # Optional filter: skip large jumps
                    if np.linalg.norm(tvec) < 3.0:
                        t_f = t_f + R_f @ tvec
                        R_f = R @ R_f
                        trajectory.append(t_f.flatten())
                        print(f"[Frame {i}] Δt = {np.linalg.norm(tvec):.3f}, Inliers = {len(inliers)}")
                    else:
                        print(f"[Frame {i}] Large motion skipped: Δt = {np.linalg.norm(tvec):.3f}")
                else:
                    print(f"[Frame {i}] solvePnPRansac failed.")
            else:
                print(f"[Frame {i}] Not enough valid 3D–2D points: {len(pts_3d)}")

        prev_img = img
        prev_kp = kp
        prev_des = des
        prev_depth = depth

    return trajectory

from tqdm import tqdm

def estimate_trajectory(left_imgs, depth_maps, K, detector, matcher, strategy, use_tqdm=True):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    R_f = np.eye(3)
    t_f = np.zeros((3, 1))
    trajectory = []

    prev_img = None
    prev_kp = None
    prev_des = None
    prev_depth = None

    print("---- Estimating trajectory ----")

    iterator = tqdm(range(len(left_imgs))) if use_tqdm else range(len(left_imgs))

    for i in iterator:
        img = cv2.imread(left_imgs[i], cv2.IMREAD_GRAYSCALE)
        depth = depth_maps[i]

        kp, des = detector.detectAndCompute(img, None)

        if prev_img is not None and prev_des is not None:
            matches = match_features(matcher, prev_des, des, strategy)

            pts_3d = []
            pts_2d = []

            for m in matches:
                u_prev, v_prev = prev_kp[m.queryIdx].pt
                u_curr, v_curr = kp[m.trainIdx].pt
                u_prev, v_prev = int(u_prev), int(v_prev)

                if 0 <= u_prev < prev_depth.shape[1] and 0 <= v_prev < prev_depth.shape[0]:
                    z = prev_depth[v_prev, u_prev]

                    if 0.1 < z < 80: # Filter bad depth
                        x = (u_prev - cx) * z / fx
                        y = (v_prev - cy) * z / fy
                        pts_3d.append([x, y, z])
                        pts_2d.append([u_curr, v_curr])
                
            # depth_vals = [pt[2] for pt in pts_3d if pt[2] > 0]
            # if len(depth_vals) > 0:
            #     print(f"[DEBUG] PnP Depth | mean: {np.mean(depth_vals):.2f}, min: {np.min(depth_vals):.2f}, max: {np.max(depth_vals):.2f}")
            # else:
            #     print(f"[WARNING] No valid depth points for frame {i}")

            if len(pts_3d) >= 6:
                pts_3d = np.array(pts_3d, dtype=np.float32)
                pts_2d = np.array(pts_2d, dtype=np.float32)
                # print(f"[DEBUG] PnP input | 3D: {pts_3d.shape}, 2D: {pts_2d.shape}")    # debug

                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    pts_3d, pts_2d, K, None, reprojectionError=8.0, flags=cv2.SOLVEPNP_ITERATIVE)

                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    t_f = t_f + R_f @ tvec
                    R_f = R @ R_f
                    trajectory.append(t_f.flatten())

                    if not use_tqdm:
                        print(f"[Frame {i}] Δt = {np.linalg.norm(tvec):.3f}, Inliers = {len(inliers)}")

                else:
                    print(f"[Frame {i}] solvePnPRansac failed.")
            else:
                print(f"[Frame {i}] Not enough valid 3D–2D points: {len(pts_3d)}")

        prev_img = img
        prev_kp = kp
        prev_des = des
        prev_depth = depth

    return trajectory
