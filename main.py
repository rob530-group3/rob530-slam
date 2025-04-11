from config import load_settings
from calibration import load_calibration, load_timestamps
from image_loader import load_image_pairs, compute_depth_maps
from feature_matcher import initialize_feature_detector
from pose_estimation import estimate_trajectory
from ground_truth import load_oxts_ground_truth, convert_to_ENU
from plot_utils import plot_trajectories, align_trajectories, compute_ate_rmse, plot_depth_map
from mapping_utils import *
from mapping import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2

DEBUG_DEPTH = False
DEBUG_MAPPING = False

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

    traj_mode = settings.get("trajectory_mode", "both")
    plot_mode = settings.get("plot_mode", "3d")
    enable_mapping = settings.get("enable_mapping", False)

    depths = compute_depth_maps(left_grey_imgs, right_grey_imgs, fx, baseline)

    if DEBUG_DEPTH:
        idx_ = 0
        plot_depth_map(depths, idx_)

    detector, matcher, strategy = initialize_feature_detector(settings)
    
    # Run VO: returns list of (R, t)
    trajectory_cam = estimate_trajectory(left_grey_imgs, depths, K, detector, matcher, strategy)

    # Extract raw camera positions
    cam_positions = np.array([t.flatten() for R, t in trajectory_cam if R is not None and t is not None])

    # Load GT and convert VO to ENU
    ground_truth = load_oxts_ground_truth(settings["oxts_folder"])
    trajectory_enu = convert_to_ENU(cam_positions)
    gt = ground_truth[:len(trajectory_enu)]

    # Raw RMSE
    raw_rmse = compute_ate_rmse(trajectory_enu, gt)
    if traj_mode in ["raw", "both"]:
        print(f"[ERROR] Raw VO RMSE: {raw_rmse:.4f} meters")

    # Umeyama alignment (anchored to origin)
    aligned_vo, scale, R_align, t_align = align_trajectories(trajectory_enu, gt, anchor_origin=True)
    aligned_rmse = compute_ate_rmse(aligned_vo, gt)
    if traj_mode in ["aligned", "both"]:
        print(f"[INFO] Alignment scale: {scale:.4f}")
        print(f"[ERROR] Aligned VO RMSE: {aligned_rmse:.4f} meters")

    # Align full trajectory (R, t)
    aligned_trajectory = []
    for R, t in trajectory_cam:
        if R is None or t is None:
            aligned_trajectory.append(None)
        else:
            R_aligned = R_align @ R
            t_aligned = scale * (R_align @ t) + t_align.reshape(3, 1)
            aligned_trajectory.append((R_aligned, t_aligned))

    # Visualization
    print("Plot mode = ", plot_mode)
    # if traj_mode == "raw":
    #     plot_trajectories(trajectory_enu, gt, title="VO (Raw) vs Ground Truth", mode=plot_mode)
    # elif traj_mode == "aligned":
    #     plot_trajectories(aligned_vo, gt, title="VO (Aligned) vs Ground Truth", mode=plot_mode)
    # elif traj_mode == "both":
    #     plot_trajectories(trajectory_enu, gt, aligned_vo=aligned_vo, title="VO (Raw + Aligned) vs Ground Truth", mode=plot_mode)

    # Mapping
    if enable_mapping:
        run_visual_mapping(
        left_colored_imgs, depths, aligned_trajectory, K, settings, DEBUG_MAPPING,
        aligned_vo=aligned_vo, gt=gt, R_align=R_align, t_align=t_align, scale=scale)

if __name__ == "__main__":
    main()
