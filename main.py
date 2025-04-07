from config import load_settings
from calibration import load_calibration, load_timestamps
from image_loader import load_image_pairs, compute_depth_maps
from feature_matcher import initialize_feature_detector
from pose_estimation import estimate_trajectory
from ground_truth import load_oxts_ground_truth, convert_to_ENU
from plot_utils import plot_trajectories, align_trajectories, compute_ate_rmse
import numpy as np
import matplotlib.pyplot as plt

DEPTH_DEBUG = False

def align_trajectories(vo, gt):
    assert vo.shape == gt.shape
    mu_vo = np.mean(vo, axis=0)
    mu_gt = np.mean(gt, axis=0)

    vo_centered = vo - mu_vo
    gt_centered = gt - mu_gt

    H = vo_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    scale = np.trace(R @ H) / np.sum(vo_centered ** 2)
    t = mu_gt.T - scale * R @ mu_vo.T

    aligned_vo = (scale * R @ vo.T).T + t.T
    return aligned_vo, scale

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
    depths = compute_depth_maps(left_imgs, right_imgs, fx, baseline)
    
    ## DEBUG:
    if DEPTH_DEBUG == True:
        depth_sample = depths[0]  # Choose any frame you want to inspect

        print("Depth map stats:")
        print("  min:", np.min(depth_sample))
        print("  max:", np.max(depth_sample))
        print("  mean:", np.mean(depth_sample))
        print("  nonzero count:", np.count_nonzero(depth_sample))

        plt.imshow(depth_sample, cmap='plasma')
        plt.colorbar()
        plt.title("Sample Depth Map")
        plt.show()


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
    print(f"[ERROR] Raw VO RMSE: {raw_rmse:.4f} meters")

    # Align VO → GT
    aligned_vo, scale = align_trajectories(trajectory_enu, gt)
    aligned_rmse = compute_ate_rmse(aligned_vo, gt)
    print(f"[INFO] Alignment scale: {scale:.4f}")   # aligned_vo = s * R * VO + t -> s = 1.00 is the best.
    print(f"[ERROR] Aligned VO RMSE: {aligned_rmse:.4f} meters")

    # Visualization mode based on settings
    mode = settings.get("trajectory_mode", "both")

    if mode == "raw":
        plot_trajectories(trajectory_enu, gt, title="VO (Raw) vs Ground Truth")
    elif mode == "aligned":
        plot_trajectories(aligned_vo, gt, title="VO (Aligned) vs Ground Truth")
    elif mode == "both":
        plot_trajectories(trajectory_enu, gt, aligned_vo=aligned_vo, title="VO (Raw + Aligned) vs Ground Truth")




if __name__ == "__main__":
    main()
