# mapping_utils.py
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

def backproject_depth_to_points(depth, K, mask=None):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()

    if mask is not None:
        valid = mask.flatten() & (z > 0)
    else:
        valid = z > 0

    u = u[valid]
    v = v[valid]
    z = z[valid]

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return np.stack((x, y, z), axis=1)

def transform_points_to_world(points_cam, R, t):
    return (R @ points_cam.T).T + t.reshape(1, 3)

def accumulate_point_cloud(depth_maps, vo_poses, K, depth_threshold=(0.1, 80.0), voxel_size=None):
    all_points = []
    for i, depth in enumerate(depth_maps):
        valid_mask = (depth > depth_threshold[0]) & (depth < depth_threshold[1])
        points_cam = backproject_depth_to_points(depth, K, mask=valid_mask)
        if points_cam.shape[0] == 0:
            continue

        R, t = vo_poses[i]
        points_world = transform_points_to_world(points_cam, R, t)
        all_points.append(points_world)

    if len(all_points) == 0:
        return np.empty((0, 3))

    global_points = np.concatenate(all_points, axis=0)
    
    if voxel_size is not None:
        global_points = voxel_downsample(global_points, voxel_size)

    return global_points

def voxel_downsample(points, voxel_size):
    discretized = np.floor(points / voxel_size)
    voxel_dict = {}
    for p, voxel in zip(points, discretized):
        key = tuple(voxel.astype(np.int32))
        if key in voxel_dict:
            voxel_dict[key].append(p)
        else:
            voxel_dict[key] = [p]
    downsampled_points = [np.mean(pts, axis=0) for pts in voxel_dict.values()]
    return np.array(downsampled_points)

def create_colored_point_cloud(img, depth_map, K, pose):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    h, w = depth_map.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map

    valid = z > 0
    x = (u[valid] - cx) * z[valid] / fx
    y = (v[valid] - cy) * z[valid] / fy
    points = np.stack((x, y, z[valid]), axis=1)

    # Color: grayscale or color image
    if len(img.shape) == 2:  # grayscale
        intensity = img[valid]
        colors = np.stack([intensity, intensity, intensity], axis=1) / 255.0
    else:  # RGB
        colors = img[valid] / 255.0

    # Transform to world frame
    points_h = np.hstack([points, np.ones((points.shape[0], 1))])
    world_points = (pose @ points_h.T).T[:, :3]

    # Create colored point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

# def visualize_point_clouds(pcd_list):
#     o3d.visualization.webrtc_server.enable_webrtc()
#     o3d.visualization.draw(pcd_list)
    
def visualize_colored_point_clouds(pcd_list):
    o3d.visualization.webrtc_server.enable_webrtc()
    o3d.visualization.draw(pcd_list)
    

def generate_point_colors_from_image(image, depth, K, mask=None):
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    H, W = depth.shape

    assert image.shape[:2] == depth.shape, "Image and depth map must match in shape"

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    u = u.flatten()
    v = v.flatten()
    z = depth.flatten()

    if mask is not None:
        valid = mask.flatten() & (z > 0)
    else:
        valid = z > 0

    u = u[valid]
    v = v[valid]

    # Clip to avoid out-of-bounds access
    u = np.clip(u, 0, W - 1)
    v = np.clip(v, 0, H - 1)

    if image.ndim == 2:
        colors = image[v, u]
        colors = np.stack([colors] * 3, axis=1)  # grayscale to RGB
    elif image.ndim == 3 and image.shape[2] == 3:
        colors = image[v, u]
    else:
        raise ValueError("Unsupported image shape.")

    if colors.dtype == np.uint8:
        colors = colors.astype(np.float32) / 255.0

    return colors

def voxel_downsample_point_cloud(pcd, voxel_size=0.2):
    return pcd.voxel_down_sample(voxel_size)

def remove_statistical_outliers(pcd, nb_neighbors=20, std_ratio=2.0):
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)


def plot_topdown_map(pcd, mode="height", aligned_vo=None, gt=None):
    """
    Visualize a 2D top-down map from a 3D point cloud with optional trajectory overlays.

    Args:
        pcd (o3d.geometry.PointCloud): Input colored point cloud.
        mode (str): Visualization mode - 'height', 'rgb', or 'density'.
        aligned_vo (np.ndarray): Optional aligned VO trajectory (Nx3).
        gt (np.ndarray): Optional ground truth trajectory (Nx3).
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        print("[WARN] Point cloud is empty.")
        return

    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(8, 6))

    if mode == "height":
        z = points[:, 2]
        sc = plt.scatter(x, y, c=z, cmap='viridis', s=1)
        plt.colorbar(sc, label="Height (Z)")

    elif mode == "rgb":
        colors = np.asarray(pcd.colors)
        # Clip RGB values to [0,1] to avoid ValueErrors
        colors = np.clip(colors, 0.0, 1.0)

        # Only keep points that have valid color entries
        if len(colors) != len(x):
            print("[WARN] Color count does not match point count — skipping RGB plot.")
        else:
            plt.scatter(x, y, c=colors, s=1)

    elif mode == "density":
        hist, xedges, yedges = np.histogram2d(x, y, bins=500)
        plt.imshow(
            hist.T,
            origin='lower',
            extent=[x.min(), x.max(), y.min(), y.max()],
            cmap='hot',
            aspect='auto'
        )
        plt.colorbar(label='Point Density')

    else:
        raise ValueError("Invalid mode. Choose from 'height', 'rgb', or 'density'.")

    # Optional trajectory overlays
    if aligned_vo is not None:
        plt.plot(aligned_vo[:, 0], aligned_vo[:, 1], 'c-', label='Aligned VO', linewidth=2)

    if gt is not None:
        plt.plot(gt[:, 0], gt[:, 1], 'r--', label='Ground Truth', linewidth=2)

    if aligned_vo is not None or gt is not None:
        plt.legend()

    plt.title(f"Top-Down 2D Map (Mode: {mode})")
    plt.xlabel("X (East)")
    plt.ylabel("Y (North)")
    plt.axis('equal')
    plt.grid(True)
    plt.show()
