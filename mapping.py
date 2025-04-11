import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm
from mapping_utils import *

def run_visual_mapping(left_colored_imgs, depths, trajectory_cam, K, settings,
                       debug=False, aligned_vo=None, gt=None, R_align=None, t_align=None, scale=None):
    
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

    # ----- Full map -----
    frame_interval = int(settings.get("frame_interval", 20))
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

        # Generate raw colored point cloud in local camera frame
        pcd = create_colored_point_cloud(img_color, depth, K, pose)
        if R_align is not None and t_align is not None and scale is not None:
            pts = np.asarray(pcd.points)
            aligned_pts = scale * (R_align @ pts.T).T + t_align.reshape(1, 3)
            pcd.points = o3d.utility.Vector3dVector(aligned_pts)

        pcd = voxel_downsample_point_cloud(pcd, voxel_size=0.2)
        pcd = remove_statistical_outliers(pcd)
        colored_pcds.append(pcd)

        if i % frame_interval == 0:
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            frame.transform(pose)
            cam_frames.append(frame)

    print(f"[INFO] Visualizing full map from {len(colored_pcds)} frames...")

    plot_mode = settings.get("plot_mode", "3d")
    if plot_mode == "2d":
        map_mode = settings.get("map_mode", "height")
        print("2D map mode = ", map_mode)
        pcd_combined = o3d.geometry.PointCloud()
        for p in colored_pcds:
            pcd_combined += p
        # print(np.asarray(pcd_combined.points)) # debug
        plot_topdown_map(pcd_combined, mode=map_mode, aligned_vo=aligned_vo, gt=gt)
    elif plot_mode == "3d":
        visualize_colored_point_clouds(colored_pcds + cam_frames)
