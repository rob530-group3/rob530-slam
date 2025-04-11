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
    enable_mapping = settings.get("enable_mapping", "False").strip().lower() == "true"

    depths = compute_depth_maps(left_grey_imgs, right_grey_imgs, fx, baseline)

    if DEBUG_DEPTH:
        idx_ = 0
        plot_depth_map(depths, idx_)

    detector, matcher, strategy = initialize_feature_detector(settings)
    
    # Run Visual Odometry - returns list of (R, t)
    raw_trajectory = estimate_trajectory(left_grey_imgs, depths, K, detector, matcher, strategy)
    
    # Extract raw camera positions (t only) from trajectory
    raw_vo = np.array([t.flatten() for R, t in raw_trajectory if R is not None and t is not None])

    # Load ground truth and convert VO to ENU coordinates
    ground_truth = load_oxts_ground_truth(settings["oxts_folder"])
    gt = ground_truth[:len(raw_vo)]

    # Evaluate raw trajectory
    raw_rmse = compute_ate_rmse(raw_vo, gt)
    if traj_mode in ["raw", "both"]:
        print(f"[ERROR] Raw VO RMSE: {raw_rmse:.4f} meters")

    # Align VO trajectory to GT using Umeyama (anchored at origin)
    aligned_vo, scale, R_align, t_align = align_trajectories(raw_vo, gt, anchor_origin=True)
    aligned_rmse = compute_ate_rmse(aligned_vo, gt)
    if traj_mode in ["aligned", "both"]:
        print(f"[INFO] Alignment scale: {scale:.4f}")
        print(f"[ERROR] Aligned VO RMSE: {aligned_rmse:.4f} meters")

    # Align full SE(3) trajectory (R, t) using Umeyama transform
    aligned_trajectory = []
    for R, t in raw_trajectory:
        if R is None or t is None:
            aligned_trajectory.append(None)
        else:
            # --- Transform Explanation ---
            # R_align: Rotation from raw VO to GT-aligned frame (from Umeyama)
            # t_align: Translation from raw VO to GT-aligned frame (from Umeyama)
            # scale: Scale factor from Umeyama (usually close to 1)
            #
            # R_aligned: rotate original camera orientation into aligned frame
            # t_aligned: scale + rotate + shift camera center into GT-aligned space
            R_aligned = R_align @ R
            t_aligned = scale * (R_align @ t) + t_align.reshape(3, 1)
            aligned_trajectory.append((R_aligned, t_aligned))
    
    # DEBUG #
    
    aligned_positions_from_trajectory = np.array([
        t.flatten() for R, t in aligned_trajectory if R is not None and t is not None
    ])

    diff = np.linalg.norm(aligned_positions_from_trajectory - aligned_vo, axis=1)
    print("[DEBUG] Mean error between aligned_vo and aligned_trajectory translation:", np.mean(diff))
    print("[DEBUG] Max  error between aligned_vo and aligned_trajectory translation:", np.max(diff))
    
    print("raw_vo[0]:", raw_vo[0])
    print("raw_vo[0]:", raw_vo[0])
    print("t from raw_trajectory[0]:", raw_trajectory[0][1].flatten())
    
    # Next debug #
    print("[DEBUG] First pose (R):\n", aligned_trajectory[0][0])
    print("[DEBUG] First pose (t):", aligned_trajectory[0][1].flatten())



    ####

    # Trajectory Plot
    print("Plot mode = ", plot_mode)
    if traj_mode == "raw":
        plot_trajectories(raw_vo, gt, title="VO (Raw) vs Ground Truth", mode=plot_mode)
    elif traj_mode == "aligned":
        plot_trajectories(aligned_vo, gt, title="VO (Aligned) vs Ground Truth", mode=plot_mode)
    elif traj_mode == "both":
        plot_trajectories(raw_vo, gt, aligned_vo=aligned_vo, title="VO (Raw + Aligned) vs Ground Truth", mode=plot_mode)

    # 3D Mapping with optional 2D projection
    if enable_mapping:
        run_visual_mapping(
            left_colored_imgs, depths, aligned_trajectory, K, settings, DEBUG_MAPPING,
            aligned_vo=aligned_vo, gt=gt
        )

if __name__ == "__main__":
    main()
