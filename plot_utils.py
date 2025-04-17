
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import open3d as o3d

def plot_trajectories(vo, gt, aligned_vo=None, title="Trajectory", mode="3d", anchor_origin=True):
    if anchor_origin:
        viz_vo = vo - vo[0]
        viz_gt = gt - gt[0]
        viz_aligned_vo = aligned_vo - aligned_vo[0] if aligned_vo is not None else None
    else:
        viz_vo = vo
        viz_gt = gt
        viz_aligned_vo = aligned_vo

    if mode == "3d":
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot(viz_vo[:, 0], viz_vo[:, 1], viz_vo[:, 2], 'o-', label="VO Trajectory")
        if viz_aligned_vo is not None:
            ax.plot(viz_aligned_vo[:, 0], viz_aligned_vo[:, 1], viz_aligned_vo[:, 2], 'x--', label="Aligned VO Trajectory")
        ax.plot(viz_gt[:, 0], viz_gt[:, 1], viz_gt[:, 2], 'r', label="Ground Truth")

        ax.set_zlabel("Z (Up)")

    elif mode == "2d":
        fig, ax = plt.subplots()

        ax.plot(viz_vo[:, 0], viz_vo[:, 1], 'o-', label="VO Trajectory")
        if viz_aligned_vo is not None:
            ax.plot(viz_aligned_vo[:, 0], viz_aligned_vo[:, 1], 'x--', label="Aligned VO Trajectory")
        ax.plot(viz_gt[:, 0], viz_gt[:, 1], 'r', label="Ground Truth")
    else:
        raise ValueError("Invalid mode. Choose '2d' or '3d'.")

    ax.set_title(title)
    ax.set_xlabel("X (East)")
    ax.set_ylabel("Y (North)")
    ax.legend()
    plt.show()
    
def align_trajectories(vo, gt, anchor_origin=True):
    # Umeyama algorithm with optional origin anchoring
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

    if anchor_origin:
        # Anchor aligned VO's starting point to origin
        t = -scale * R @ vo[0].T
    else:
        t = mu_gt.T - scale * R @ mu_vo.T

    aligned_vo = (scale * R @ vo.T).T + t.T
    return aligned_vo, scale, R, t

def compute_ate_rmse(vo, gt):
    assert vo.shape == gt.shape
    return np.sqrt(np.mean(np.sum((vo - gt) ** 2, axis=1)))

def plot_depth_map(depths, i):
    if i >= len(depths):
        print("Index out of range")
        return
    
    depth_sample = depths[i]  

    print("Depth map stats:")
    print("  min:", np.min(depth_sample))
    print("  max:", np.max(depth_sample))
    print("  mean:", np.mean(depth_sample))
    print("  nonzero count:", np.count_nonzero(depth_sample))

    plt.imshow(depth_sample, cmap='plasma')
    plt.colorbar()
    plt.title("Sample Depth Map")
    plt.show()
    
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
    

