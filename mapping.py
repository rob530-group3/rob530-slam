import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from mapping_utils import *

def run_visual_mapping(left_colored_imgs, depths, aligned_trajectory, K, settings,
                       debug=False, aligned_vo=None, gt=None, R_align=None, t_align=None, scale=None):
    """
    Visualizes a 3D or 2D map reconstructed from aligned visual odometry trajectory.

    Args:
        left_colored_imgs (list): List of left camera RGB images.
        depths (list): List of depth maps.
        aligned_trajectory (list): List of (R, t) aligned poses.
        K (ndarray): Camera intrinsics.
        settings (dict): User configuration dictionary.
        debug (bool): If True, only show one frame for inspection.
        aligned_vo (ndarray): Aligned trajectory positions for overlay (optional).
        gt (ndarray): Ground truth trajectory for overlay (optional).
        R_align, t_align, scale: Umeyama transformation components.
    """

    if debug:
        i = 0
        img_color = cv2.cvtColor(left_colored_imgs[i], cv2.COLOR_BGR2RGB)
        depth = depths[i]
        R_aligned, t_aligned = aligned_trajectory[i]
        
        R_correction = np.array([[-1, 0, 0], [0, -1, 0], [0,0,-1]])     # fix the camera backward orientation
        R_aligned = R_correction @ R_aligned

        pose = np.eye(4)
        pose[:3, :3] = R_aligned
        pose[:3, 3] = t_aligned.squeeze()

        # Generate point cloud in aligned world frame
        pcd = create_colored_point_cloud(img_color, depth, K, pose)
        visualize_colored_point_clouds(pcd)
        return

    frame_interval = int(settings.get("frame_interval", 20))
    colored_pcds = []
    cam_frames = []

    print("---- Constructing map ----")

    for i in tqdm(range(len(aligned_trajectory))):
        if aligned_trajectory[i] is None:
            print(f"[Frame {i}] Pose is None. Skipping.")
            continue

        img_color = cv2.cvtColor(left_colored_imgs[i], cv2.COLOR_BGR2RGB)
        depth = depths[i]

        # # Transform each pose using Umeyama to global frame
        R_aligned, t_aligned = aligned_trajectory[i]
        
        R_correction = np.array([[-1, 0, 0], [0, -1, 0], [0,0,1]])     # fix the camera backward orientation
        R_aligned = R_correction @ R_aligned

        pose = np.eye(4)
        pose[:3, :3] = R_aligned
        pose[:3, 3] = t_aligned.squeeze()

        # Generate point cloud in aligned world coordinates
        pcd = create_colored_point_cloud(img_color, depth, K, pose)
        pcd = voxel_downsample_point_cloud(pcd, voxel_size=0.2)
        pcd = remove_statistical_outliers(pcd)
        colored_pcds.append(pcd)

        # Add coordinate frame at this pose every N frames
        if i % frame_interval == 0:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            frame.transform(pose)
            cam_frames.append(frame)

    print(f"[INFO] Visualizing full map from {len(colored_pcds)} frames...")

    plot_mode = settings.get("plot_mode", "3d")
    if plot_mode == "2d":
        map_mode = settings.get("map_mode", "height")
        print("2D map mode =", map_mode)

        # Merge point clouds and convert mesh frames to sampled points
        pcd_combined = o3d.geometry.PointCloud()
        for pcd in colored_pcds + cam_frames:
            if isinstance(pcd, o3d.geometry.PointCloud):
                pcd_combined += pcd
            elif isinstance(pcd, o3d.geometry.TriangleMesh):
                pcd_combined += pcd.sample_points_uniformly(number_of_points=50)

        # # debug
        points = np.asarray(pcd_combined.points)
        print("[DEBUG] Map point cloud stats:")
        print("  X range:", np.min(points[:, 0]), "to", np.max(points[:, 0]))
        print("  Y range:", np.min(points[:, 1]), "to", np.max(points[:, 1]))
        print("  Z range:", np.min(points[:, 2]), "to", np.max(points[:, 2]))
        print("  Total points:", points.shape[0])
        
        # if aligned_vo is not None:
        #     print("[DEBUG] Aligned VO range:")
        #     print("  X:", np.min(aligned_vo[:, 0]), "to", np.max(aligned_vo[:, 0]))
        #     print("  Y:", np.min(aligned_vo[:, 1]), "to", np.max(aligned_vo[:, 1]))
        
        # if aligned_vo is not None:
        #     centroid_map = np.mean(points, axis=0)
        #     centroid_vo = np.mean(aligned_vo, axis=0)
        #     offset = np.linalg.norm(centroid_map[:2] - centroid_vo[:2])
        #     print("[DEBUG] Centroid 2D offset between map and VO:", offset)
        # ##########################

        # Plot top-down 2D projection with optional trajectory overlays
        plot_topdown_map(pcd_combined, mode=map_mode, aligned_vo=aligned_vo, gt=gt)

    elif plot_mode == "3d":
        visualize_colored_point_clouds(colored_pcds + cam_frames)
