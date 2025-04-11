
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from mapping_utils import create_colored_point_cloud, voxel_downsample_point_cloud, remove_statistical_outliers, visualize_colored_point_clouds

def run_visual_mapping(left_colored_imgs, depths, trajectory_cam, K, settings, debug=False):
    if debug:
        i = 0  # Frame index for test

        img_color = cv2.cvtColor(left_colored_imgs[i], cv2.COLOR_BGR2RGB)
        depth = depths[i]

        R, t = trajectory_cam[i]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.squeeze()

        pcd = create_colored_point_cloud(img_color, depth, K, pose)
        visualize_colored_point_clouds(pcd)

    # ----- Construct full colored map with filtering -----
    frame_interval = settings.get("frame_inteval", 20)
    colored_pcds = []
    cam_frames = []

    print("---- Constructing map ----")

    for i in tqdm(range(len(trajectory_cam))):
        if trajectory_cam[i] is None:
            print(f"[Frame {i}] Pose is None. Skipping.")
            continue

        img_color = cv2.cvtColor(left_colored_imgs[i], cv2.COLOR_BGR2RGB)
        depth = depths[i]

        R, t = trajectory_cam[i]
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = t.squeeze()

        # Generate point cloud
        pcd = create_colored_point_cloud(img_color, depth, K, pose)
        pcd = voxel_downsample_point_cloud(pcd, voxel_size=0.2)
        pcd = remove_statistical_outliers(pcd)
        colored_pcds.append(pcd)

        # Add coordinate frame every N frames
        if i % frame_interval == 0:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            frame.transform(pose)
            cam_frames.append(frame)

    print(f"[INFO] Visualizing full map from {len(colored_pcds)} frames...")
    visualize_colored_point_clouds(colored_pcds + cam_frames)
