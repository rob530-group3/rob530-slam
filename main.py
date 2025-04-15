from config import load_settings
from calibration import load_calibration
from image_loader import load_image_pairs, compute_depth_maps
from feature_matcher import initialize_feature_detector
from pose_estimation import initialize_vo_state, estimate_vo_step
from ground_truth import load_oxts_ground_truth
from plot_utils import plot_trajectories, align_trajectories, compute_ate_rmse, plot_depth_map
from mapping import run_visual_mapping
from mapping_utils import plot_topdown_map, visualize_colored_point_clouds, apply_umeyama_to_pointclouds
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d as o3d

DEBUG_DEPTH = False
DEBUG_MAPPING = False

def merge_pointclouds_with_frames(pcd_list, cam_frames):
    """
    Merge a list of point clouds and add coordinate frames (converted to point clouds).

    Args:
        pcd_list (list of o3d.geometry.PointCloud): List of point clouds.
        cam_frames (list of o3d.geometry.TriangleMesh): List of camera frame meshes.

    Returns:
        o3d.geometry.PointCloud: Merged point cloud including frames.
    """
    merged = o3d.geometry.PointCloud()
    for pcd in pcd_list:
        merged += pcd
    for frame in cam_frames:
        sampled_frame = frame.sample_points_uniformly(number_of_points=50)
        merged += sampled_frame
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
    raw_vo_positions = np.array(raw_vo_positions)
    ground_truth = load_oxts_ground_truth(settings["oxts_folder"])
    gt = ground_truth[:len(raw_vo_positions)]

    aligned_vo, scale, R_align, t_align = align_trajectories(raw_vo_positions, gt, anchor_origin=True)
    aligned_rmse = compute_ate_rmse(aligned_vo, gt)
    print(f"[INFO] Aligned VO RMSE: {aligned_rmse:.4f}")

    # Step 4: Apply Umeyama to point cloud map
    aligned_map = apply_umeyama_to_pointclouds(pointclouds, R_align, t_align, scale)

    # Step 5: Visualization
    merged_map = merge_pointclouds_with_frames(aligned_map)

    plot_mode = settings.get("plot_mode", "3d")
    if plot_mode == "2d":
        map_mode = settings.get("map_mode", "height")
        print("2D map mode =", map_mode)

        # Merge point clouds and convert mesh frames to sampled points
        merged_map = merge_pointclouds_with_frames(aligned_map, cam_frames)
        plot_topdown_map(merged_map, mode=map_mode, aligned_vo=aligned_vo, gt=gt)
    elif plot_mode == "3d":
        visualize_colored_point_clouds(aligned_map + cam_frames)

if __name__ == "__main__":
    main()
