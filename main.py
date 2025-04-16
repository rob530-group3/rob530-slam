from config import load_settings
from calibration import load_calibration
from image_loader import load_image_pairs, compute_depth_maps
from feature_matcher import initialize_feature_detector
from pose_estimation import initialize_vo_state, estimate_vo_step
from ground_truth import load_oxts_ground_truth
from plot_utils import plot_trajectories, align_trajectories, compute_ate_rmse, plot_depth_map
from mapping import run_visual_mapping
from mapping_utils import plot_topdown_map, visualize_colored_point_clouds, apply_umeyama_to_pointclouds
from loop_closure import *
from pose_graph import *
from gtsam import symbol_shorthand  # <- Add this
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
import gtsam

DEBUG_DEPTH = False
DEBUG_MAPPING = False

def merge_pointclouds_with_frames(pcd_list, cam_frames):
    """
    Merge a list of point clouds and include camera frames (already as point clouds).

    Args:
        pcd_list (list of o3d.geometry.PointCloud): Point clouds.
        cam_frames (list of o3d.geometry.PointCloud): Sampled camera frame points.

    Returns:
        o3d.geometry.PointCloud: Merged point cloud including sampled camera frames.
    """
    merged = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        merged += pcd
    for frame_pcd in cam_frames:
        merged += frame_pcd
    return merged

def main():
    settings = load_settings("settings.txt")
    K, fx, baseline = load_calibration(settings["calib_file"])

    left_grey_imgs, right_grey_imgs, left_colored_imgs, right_colored_imgs = load_image_pairs(
        settings["left_grey_folder"],
        settings["right_grey_folder"],
        settings["left_colored_folder"],
        settings["right_colored_folder"],
        settings["timestamp_file_left"],
        settings["timestamp_file_right"]
    )

    depths = compute_depth_maps(left_grey_imgs, right_grey_imgs, fx, baseline)

    if DEBUG_DEPTH:
        plot_depth_map(depths, 0)

    detector, matcher, strategy = initialize_feature_detector(settings)

    raw_trajectory = []
    raw_vo_positions = []
    pointclouds = []
    cam_frames = []

    # Step 1: Initialize VO
    prev_img, prev_kp, prev_des, prev_depth = initialize_vo_state(left_grey_imgs[0], depths[0], detector)
    R_f = np.eye(3)
    t_f = np.zeros((3, 1))
    raw_trajectory.append((R_f.copy(), t_f.copy()))
    raw_vo_positions.append(t_f.flatten())
    
    loop_detector = LoopClosureDetector(detector, matcher)
    loop_constraints = []
    
    X = symbol_shorthand.X   

    # Step 2: SLAM loop
    print("---- Running SLAM loop ----")
    for i in tqdm(range(1, len(left_grey_imgs))):
        curr_img_path = left_grey_imgs[i]
        curr_depth = depths[i]

        prev_img, prev_kp, prev_des, prev_depth, pose = estimate_vo_step(
            curr_img_path, curr_depth,
            prev_img, prev_kp, prev_des, prev_depth,
            K, detector, matcher, strategy,
            R_f, t_f
        )

        if pose is None:
            raw_trajectory.append(None)
            continue
        
        loop_detector.add_keyframe(prev_img)  # This should already be there

        matched_idx = loop_detector.detect_loop(prev_img, i)
        if matched_idx is not None:
            # print(f"[LOOP CLOSURE] Frame {i} matched with Frame {matched_idx}")

            # Use keypoints and descriptors to estimate relative pose between i and matched_idx
            img_i, kp_i, des_i = loop_detector.keyframes[-1]
            img_j, kp_j, des_j = loop_detector.keyframes[matched_idx]

            # Match descriptors (recompute if necessary)
            matches = matcher.knnMatch(des_i, des_j, k=2)
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            # Prepare 3D-2D correspondences for PnP
            pts_3d = []
            pts_2d = []
            for match in good_matches:
                u_i, v_i = kp_i[match.queryIdx].pt
                u_j, v_j = kp_j[match.trainIdx].pt
                u_i, v_i = int(u_i), int(v_i)
                
                if 0 <= u_i < depths[i].shape[1] and 0 <= v_i < depths[i].shape[0]:
                    z = depths[i][v_i, u_i]
                    if z <= 0:
                        continue
                    x = (u_i - K[0, 2]) * z / K[0, 0]
                    y = (v_i - K[1, 2]) * z / K[1, 1]
                    pts_3d.append([x, y, z])
                    pts_2d.append([u_j, v_j])

            if len(pts_3d) >= 6:
                pts_3d = np.array(pts_3d, dtype=np.float32)
                pts_2d = np.array(pts_2d, dtype=np.float32)
                success, rvec, tvec, _ = cv2.solvePnPRansac(pts_3d, pts_2d, K, None)

                if success:
                    R_rel, _ = cv2.Rodrigues(rvec)
                    t_rel = tvec
                    constraint = LoopClosureConstraint(i, matched_idx, (R_rel, t_rel))
                    loop_constraints.append(constraint)
                    print(f"[INFO] Loop constraint between {i} and {matched_idx} added. Total: {len(loop_constraints)}")

        R_f, t_f = pose
        raw_trajectory.append((R_f.copy(), t_f.copy()))
        raw_vo_positions.append(t_f.flatten())

        img_color = cv2.cvtColor(left_colored_imgs[i], cv2.COLOR_BGR2RGB)
        pcd = run_visual_mapping(img_color, curr_depth, K, pose)
        pointclouds.append(pcd)
        
        # Add coordinate frame every N frames
        frame_interval = int(settings.get("frame_interval", 20))
        if i % frame_interval == 0:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            frame.transform(np.vstack((np.hstack((R_f, t_f)), [0, 0, 0, 1])))  # pose = [R | t]
            cam_frames.append(frame)

    # Step 3: Trajectory alignment
    print("[INFO] Running pose graph optimization...")
    optimized_values = build_pose_graph(raw_trajectory, loop_constraints)
    print("[INFO] Optimization complete.")

    optimized_trajectory = []
    raw_vo_positions = []

    for i in range(len(raw_trajectory)):
        key = X(i)
        if optimized_values.exists(key):
            pose_i = optimized_values.atPose3(key)
            R = pose_i.rotation().matrix()
            t = pose_i.translation()
            optimized_trajectory.append((R, t))
            raw_vo_positions.append(t.flatten())
        else:
            optimized_trajectory.append(None)  # preserve indexing

    raw_vo_positions = np.array(raw_vo_positions)

    ground_truth = load_oxts_ground_truth(settings["oxts_folder"])
    gt = ground_truth[:len(raw_vo_positions)]

    aligned_vo, scale, R_align, t_align = align_trajectories(raw_vo_positions, gt, anchor_origin=True)
    aligned_rmse = compute_ate_rmse(aligned_vo, gt)
    print(f"[INFO] Aligned VO RMSE: {aligned_rmse:.4f}")

    # Step 4: Apply Umeyama to point cloud map
    aligned_map = apply_umeyama_to_pointclouds(pointclouds, R_align, t_align, scale)
    aligned_frames = []
    for frame in cam_frames:
        sampled = frame.sample_points_uniformly(number_of_points=50)
        pts = np.asarray(sampled.points)
        pts_aligned = scale * (R_align @ pts.T).T + t_align.reshape(1, 3)
        sampled.points = o3d.utility.Vector3dVector(pts)
        aligned_frames.append(sampled)

    # Step 5: Visualization
    plot_mode = settings.get("plot_mode", "3d")
    if plot_mode == "2d":
        map_mode = settings.get("map_mode", "height")
        print("2D map mode =", map_mode)

        # Merge point clouds and convert mesh frames to sampled points
        merged_map = merge_pointclouds_with_frames(aligned_map, aligned_frames)
        plot_topdown_map(merged_map, mode=map_mode, aligned_vo=aligned_vo, gt=gt)
    elif plot_mode == "3d":
        visualize_colored_point_clouds(aligned_map + cam_frames)

if __name__ == "__main__":
    main()
