from config import load_settings
from calibration import load_calibration, load_timestamps
from image_loader import load_image_pairs, compute_depth_maps
from feature_matcher import initialize_feature_detector
from pose_estimation import estimate_trajectory
from ground_truth import load_oxts_ground_truth, convert_to_ENU
from plot_utils import plot_trajectories, align_trajectories, compute_ate_rmse, plot_depth_map
import numpy as np
import matplotlib.pyplot as plt

DEPTH_DEBUG = False

def main():
    # Load settings from config file
    settings = load_settings("settings.txt")

    # Load calibration and timestamps
    K, fx, baseline = load_calibration(settings["calib_file"])

    # Load image pairs and compute depth maps
    left_imgs, right_imgs = load_image_pairs(
    settings["left_folder"],
    settings["right_folder"],
    settings["timestamp_file_left"],
    settings["timestamp_file_right"]
    )
    
    # Visualization setting
    traj_mode = settings.get("trajectory_mode", "both")
    plot_mode = settings.get("plot_mode", "3d").strip() # 2D or 3D plot mode
    
    depths = compute_depth_maps(left_imgs, right_imgs, fx, baseline)
    
    ## DEBUG:
    if DEPTH_DEBUG == True:
        idx_= 0 # Choose any frame you want to inspect
        plot_depth_map(depths, idx_)

    # Initialize feature extractor and matcher
    detector, matcher, strategy = initialize_feature_detector(settings)

    # Run visual odometry
    trajectory_cam = estimate_trajectory(left_imgs, depths, K, detector, matcher, strategy)

    # Load and process ground truth from OXTS
    ground_truth = load_oxts_ground_truth(settings["oxts_folder"])

    # Convert to ENU
    trajectory_enu = convert_to_ENU(np.array(trajectory_cam))
    gt = ground_truth[:len(trajectory_enu)]

    # Compute RMSE for raw VO
    raw_rmse = compute_ate_rmse(trajectory_enu, gt)
    if traj_mode == "raw" or "both":
        print(f"[ERROR] Raw VO RMSE: {raw_rmse:.4f} meters")

    # Align VO → GT
    aligned_vo, scale = align_trajectories(trajectory_enu, gt)
    aligned_rmse = compute_ate_rmse(aligned_vo, gt)
    if traj_mode == "aligned" or "both":
        print(f"[INFO] Alignment scale: {scale:.4f}")   # aligned_vo = s * R * VO + t -> s = 1.00 is the best.
        print(f"[ERROR] Aligned VO RMSE: {aligned_rmse:.4f} meters")

    # Visualization mode based on settings
    print("Plot mode = ", plot_mode)
    
    if traj_mode == "raw":
        plot_trajectories(trajectory_enu, gt, title="VO (Raw) vs Ground Truth", mode=plot_mode)
    elif traj_mode == "aligned":
        plot_trajectories(aligned_vo, gt, title="VO (Aligned) vs Ground Truth", mode=plot_mode)
    elif traj_mode == "both":
        plot_trajectories(trajectory_enu, gt, aligned_vo=aligned_vo, title="VO (Raw + Aligned) vs Ground Truth", mode=plot_mode)
    
    # # debug
    # vo_np = np.array(trajectory_enu)  # ENU coords (same as aligned)
    # gt_np = np.array(ground_truth[:len(vo_np)])

    # # Plot Z component
    # plt.figure()
    # plt.plot(vo_np[:, 2], label="VO Z (Up)")
    # plt.plot(gt_np[:, 2], label="GT Z (Up)")
    # plt.title("Z-axis (Vertical) Drift")
    # plt.xlabel("Frame")
    # plt.ylabel("Z (Up)")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
