
import os
import cv2
import glob
import numpy as np
from tqdm import tqdm
from calibration import load_timestamps

def sync_stereo_indices(left_ts, right_ts, max_diff=0.02):
    """
    Finds index pairs of left and right images whose timestamps are closest within `max_diff` seconds.
    """
    matched_indices = []
    r_idx = 0

    for l_idx, l_time in enumerate(left_ts):
        # Find the best right timestamp match for this left timestamp
        while r_idx + 1 < len(right_ts) and abs(right_ts[r_idx + 1] - l_time) < abs(right_ts[r_idx] - l_time):
            r_idx += 1

        if abs(right_ts[r_idx] - l_time) <= max_diff:
            matched_indices.append((l_idx, r_idx))

    return matched_indices

def load_image_pairs(left_folder, right_folder, left_ts_file, right_ts_file):
    left_ts = load_timestamps(left_ts_file)
    right_ts = load_timestamps(right_ts_file)
    matched_indices = sync_stereo_indices(left_ts, right_ts)

    left_images = sorted(os.listdir(left_folder))
    right_images = sorted(os.listdir(right_folder))

    left_paths = []
    right_paths = []

    print("---- Loading images ----")
    for l_idx, r_idx in tqdm(list(matched_indices)):
        left_path = os.path.join(left_folder, left_images[l_idx])
        right_path = os.path.join(right_folder, right_images[r_idx])

        if os.path.exists(left_path) and os.path.exists(right_path):
            left_paths.append(left_path)
            right_paths.append(right_path)

    return left_paths, right_paths

def compute_depth_maps(left_imgs, right_imgs, fx, baseline):
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=11,
        P1=8*3*11**2, P2=32*3*11**2,
        disp12MaxDiff=1, uniquenessRatio=10,
        speckleWindowSize=100, speckleRange=32)
    
    depth_maps = []
    print("---- Computing depths ----")
    for left, right in tqdm(zip(left_imgs, right_imgs)):
        imgL = cv2.imread(left, cv2.IMREAD_GRAYSCALE)
        imgR = cv2.imread(right, cv2.IMREAD_GRAYSCALE)
        disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
        depth = np.zeros_like(disparity)
        valid = disparity > 0
        depth[valid] = (fx * baseline) / disparity[valid]
        depth_maps.append(depth)
    return depth_maps
