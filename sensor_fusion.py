from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

def lidar_image_overlay_pipeline(
    frame_id,
    image_path,
    lidar_path,
    calib_cam_path,
    calib_velo_path,
    cam_id="02",
    show=True
):

    def read_calib_file(filepath):
        calib = {}
        with open(filepath, 'r') as f:
            for line in f:
                if ':' not in line:
                    continue
                key, value = line.strip().split(':', 1)
                try:
                    calib[key] = np.array([float(x) for x in value.strip().split()])
                except ValueError:
                    continue
        for k in calib:
            if calib[k].size == 12:
                calib[k] = calib[k].reshape(3, 4)
            elif calib[k].size == 9:
                calib[k] = calib[k].reshape(3, 3)
            elif calib[k].size == 3:
                calib[k] = calib[k].reshape(3,)
        return calib

    def get_projection_matrix(calib_cam_to_cam, calib_velo_to_cam, cam='02'):
        P_rect = calib_cam_to_cam[f'P_rect_{cam}']
        R_rect = np.eye(4)
        R_rect[:3, :3] = calib_cam_to_cam[f'R_rect_{cam}']
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[:3, :3] = calib_velo_to_cam['R'].reshape(3, 3)
        Tr_velo_to_cam[:3, 3] = calib_velo_to_cam['T']
        proj_mat = P_rect @ R_rect @ Tr_velo_to_cam
        return proj_mat

    def project_lidar_to_image(points, proj_mat, image_shape):
        h, w = image_shape
        depth_map = np.zeros((h, w), dtype=np.float32)

        points_hom = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
        pts_2d_hom = (proj_mat @ points_hom.T).T
        pts_2d = pts_2d_hom[:, :2] / pts_2d_hom[:, 2:3]
        depth = pts_2d_hom[:, 2]

        x_img = np.round(pts_2d[:, 0]).astype(int)
        y_img = np.round(pts_2d[:, 1]).astype(int)

        valid = (depth > 0) & (x_img >= 0) & (x_img < w) & (y_img >= 0) & (y_img < h)

        pts_2d_valid = pts_2d[valid]
        depth_valid = depth[valid]
        x_valid = x_img[valid]
        y_valid = y_img[valid]

        for x, y, d in zip(x_valid, y_valid, depth_valid):
            if depth_map[y, x] == 0 or depth_map[y, x] > d:
                depth_map[y, x] = d

        return pts_2d_valid, depth_valid, depth_map

    def overlay_lidar_on_image(image, pts_2d, depths):
        img = image.copy()
        norm_depths = (depths - depths.min()) / (depths.ptp() + 1e-5)
        cmap = plt.cm.get_cmap('jet')
        for i, (x, y) in enumerate(pts_2d.astype(int)):
            color = (np.array(cmap(norm_depths[i]))[:3] * 255).astype(np.uint8)
            cv2.circle(img, (x, y), radius=1, color=tuple(int(c) for c in color), thickness=-1)
        return img

    image = np.array(Image.open(image_path).convert("RGB"))
    image_shape = image.shape[:2]
    points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
    calib_cam = read_calib_file(calib_cam_path)
    calib_velo = read_calib_file(calib_velo_path)

    proj_mat = get_projection_matrix(calib_cam, calib_velo, cam=cam_id)
    pts_2d, depths, depth_map = project_lidar_to_image(points, proj_mat, image_shape)
    overlay = overlay_lidar_on_image(image, pts_2d, depths)

    if show:
        plt.imshow(overlay)
        plt.axis('off')
        plt.title("LiDAR Projected to Image Plane")
        plt.show()

    return overlay, depth_map, pts_2d, depths


def refine_stereo_with_lidar(stereo_depth, lidar_depth, method='blend', alpha=0.5, threshold=1.0):

    assert stereo_depth.shape == lidar_depth.shape, "Shape mismatch"
    
    refined_depth = stereo_depth.copy()
    mask_lidar = lidar_depth > 0  # LiDAR 有效区域
    
    if method == 'replace':
        refined_depth[mask_lidar] = lidar_depth[mask_lidar]
    
    elif method == 'blend':

        diff = np.abs(stereo_depth - lidar_depth)
        large_diff_mask = (diff > threshold) & mask_lidar
        
        # linear interpolation
        refined_depth[large_diff_mask] = (
            alpha * stereo_depth[large_diff_mask] + 
            (1 - alpha) * lidar_depth[large_diff_mask]
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return refined_depth

def fuse_stereo_lidar_depth(stereo_depth, lidar_depth, guide_img, edge_sigma=3, conf_threshold=0.6):
    """
    高级融合：结合 lidar 和 stereo 优缺点，构造融合深度图
    
    参数：
        stereo_depth: (H, W) numpy array, 双目稠密深度
        lidar_depth:  (H, W) numpy array, 稀疏 LiDAR 深度
        guide_img:    (H, W, 3) RGB 图像, 引导边缘结构
        edge_sigma:   bilateral filter 空间平滑程度
        conf_threshold: Stereo 置信度阈值，低于此视为不可信
    
    返回：
        fused_depth: 融合后的深度图
    """
    assert stereo_depth.shape == lidar_depth.shape, "尺寸不匹配"
    H, W = stereo_depth.shape

    # Step 1: Stereo 置信度（基于梯度）
    grad_x = cv2.Sobel(stereo_depth, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(stereo_depth, cv2.CV_32F, 0, 1, ksize=5)
    gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
    stereo_conf = np.exp(-gradient_mag / (np.std(gradient_mag) + 1e-5))  # [0, 1]

    # Step 2: 双边滤波（边缘引导的平滑）
    lidar_filtered = cv2.bilateralFilter(lidar_depth.astype(np.float32), d=-1,
                                         sigmaColor=0.1*255, sigmaSpace=edge_sigma)
    stereo_filtered = cv2.bilateralFilter(stereo_depth.astype(np.float32), d=-1,
                                          sigmaColor=0.1*255, sigmaSpace=edge_sigma)

    # Step 3: 融合 mask 逻辑（可换更复杂规则）
    lidar_mask = lidar_depth > 0
    stereo_mask = stereo_conf > conf_threshold

    # Step 4: 自适应融合权重
    alpha = np.zeros_like(stereo_depth)
    alpha[lidar_mask] = 0.9  # LiDAR 有效时权重大
    alpha[~lidar_mask & stereo_mask] = 0.2  # LiDAR 无效但 stereo 稳定时保留一点 stereo
    alpha[~lidar_mask & ~stereo_mask] = 0.0  # 都不可靠区域放弃或留空

    # Step 5: 融合
    fused_depth = alpha * lidar_filtered + (1 - alpha) * stereo_filtered
    return fused_depth