import cv2
import numpy as np
import open3d as o3d
from mapping_utils import create_colored_point_cloud, voxel_downsample_point_cloud, remove_statistical_outliers

def run_visual_mapping(img_color, depth, K, pose):
    """
    Given a color image, depth map, camera intrinsics, and pose,
    generate a filtered point cloud (for one frame).

    Args:
        img_color (np.ndarray): RGB image.
        depth (np.ndarray): Depth map.
        K (np.ndarray): Camera intrinsic matrix.
        pose (np.ndarray): 4x4 transformation matrix from camera to world.

    Returns:
        o3d.geometry.PointCloud: Filtered point cloud in world frame.
    """
    # Create point cloud in camera frame
    pcd = create_colored_point_cloud(img_color, depth, K, pose)

    # Clean up point cloud
    pcd = voxel_downsample_point_cloud(pcd, voxel_size=0.2)
    pcd = remove_statistical_outliers(pcd)

    return pcd
