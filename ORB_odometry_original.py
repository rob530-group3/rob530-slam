# ORB_odometry_with_depth.py
import os
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from datetime import datetime
from pyproj import Proj

# === CONFIGURABLE PATHS ===
left_folder = "/home/phurithat/Documents/ros2_ws/raw/2011_09_26/2011_09_26_drive_0002_sync/image_00/data"
right_folder = "/home/phurithat/Documents/ros2_ws/raw/2011_09_26/2011_09_26_drive_0002_sync/image_01/data"
timestamp_file = "/home/phurithat/Documents/ros2_ws/raw/2011_09_26/2011_09_26_drive_0002_sync/image_00/timestamps.txt"
calib_file = "/home/phurithat/Documents/ros2_ws/raw/2011_09_26/2011_09_26_calib/2011_09_26/calib_cam_to_cam.txt"
oxts_folder = "/home/phurithat/Documents/ros2_ws/raw/2011_09_26/2011_09_26_drive_0002_sync/oxts/data"

# === Timestamp parser ===
def parse_timestamps(file_path):
    with open(file_path, "r") as f:
        timestamps = [line.strip() for line in f.readlines()]
    timestamps_dt = []
    for ts in timestamps:
        main_time, nano_sec = ts[:19], ts[20:]
        dt_obj = datetime.strptime(main_time, "%Y-%m-%d %H:%M:%S")
        total_seconds = (dt_obj - datetime(1970, 1, 1)).total_seconds() + float("0." + nano_sec)
        timestamps_dt.append(total_seconds)
    timestamps_dt = np.array(timestamps_dt)
    timestamps_dt -= timestamps_dt[0]  # Normalize
    return timestamps_dt

# === Calibration parser ===
def parse_calib_cam_to_cam(file_path):
    calib = {}
    with open(file_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.strip().split(':', 1)
                try:
                    calib[key.strip()] = np.array([float(x) for x in value.strip().split()])
                except ValueError:
                    continue
    K_left = calib['K_00'].reshape(3, 3)
    P_left = calib['P_rect_00'].reshape(3, 4)
    P_right = calib['P_rect_01'].reshape(3, 4)
    fx = P_left[0, 0]
    baseline = abs(P_right[0, 3] - P_left[0, 3]) / fx
    return K_left, fx, baseline

# === Load calibration and timestamps ===
K, fx, baseline = parse_calib_cam_to_cam(calib_file)
timestamps = parse_timestamps(timestamp_file)

# === Load image pairs ===
left_images = sorted(glob.glob(os.path.join(left_folder, "*.png")))
right_images = sorted(glob.glob(os.path.join(right_folder, "*.png")))

# === Stereo matcher ===
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=11,
    P1=8 * 3 * 11 ** 2,
    P2=32 * 3 * 11 ** 2,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=32
)

# === ORB and Matcher ===
orb = cv2.ORB_create(2000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

trajectory = []
R_f = np.eye(3)
t_f = np.zeros((3, 1))

prev_img = None
prev_kp = None
prev_des = None
prev_depth = None

for i in range(len(left_images)):
    img_left = cv2.imread(left_images[i], cv2.IMREAD_GRAYSCALE)
    img_right = cv2.imread(right_images[i], cv2.IMREAD_GRAYSCALE)

    # Compute disparity & depth
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    depth = np.zeros_like(disparity)
    valid = disparity > 0
    depth[valid] = (fx * baseline) / disparity[valid]

    kp, des = orb.detectAndCompute(img_left, None)

    if prev_img is not None and prev_des is not None:
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        pts_prev = np.array([prev_kp[m.queryIdx].pt for m in matches], dtype=np.float32)
        pts_curr = np.array([kp[m.trainIdx].pt for m in matches], dtype=np.float32)

        # Back-project to 3D using previous depth map
        pts_3d = []
        pts_2d = []
        for pt2d_prev, pt2d_curr in zip(pts_prev, pts_curr):
            u, v = int(pt2d_prev[0]), int(pt2d_prev[1])
            if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
                z = prev_depth[v, u]
                if z > 0:  
                    x = (u - K[0, 2]) * z / K[0, 0]
                    y = (v - K[1, 2]) * z / K[1, 1]
                    pts_3d.append([x, y, z])
                    pts_2d.append(pt2d_curr)

        if len(pts_3d) >= 6:
            pts_3d = np.array(pts_3d, dtype=np.float32)
            pts_2d = np.array(pts_2d, dtype=np.float32)
            _, rvec, tvec, inliers = cv2.solvePnPRansac(pts_3d, pts_2d, K, None)
            R, _ = cv2.Rodrigues(rvec)
            t_f = t_f + R_f @ tvec  # Compose incrementally
            R_f = R @ R_f
            trajectory.append(t_f.flatten())

    prev_img = img_left
    prev_kp = kp
    prev_des = des
    prev_depth = depth

trajectory = np.array(trajectory)

# === Load OXTS GPS Ground Truth ===
oxts_files = sorted([os.path.join(oxts_folder, f) for f in os.listdir(oxts_folder) if f.endswith(".txt")])
lat_lon_alt = []
for file in oxts_files:
    with open(file, "r") as f:
        values = list(map(float, f.readline().strip().split()))
        lat_lon_alt.append(values[:3])
lat_lon_alt = np.array(lat_lon_alt)

lat0, lon0, alt0 = lat_lon_alt[0]
proj_utm = Proj(proj="utm", zone=32, datum="WGS84")
x, y = proj_utm(lat_lon_alt[:, 1], lat_lon_alt[:, 0])
z = lat_lon_alt[:, 2] - alt0
x -= x[0]
y -= y[0]
ground_truth = np.vstack((x, y, z)).T

# === Convert VO trajectory to ENU frame ===
trajectory_ENU = np.zeros_like(trajectory)
trajectory_ENU[:, 0] = trajectory[:, 2]    # Z_cam → X_enu
trajectory_ENU[:, 1] = -trajectory[:, 0]   # X_cam → Y_enu
trajectory_ENU[:, 2] = -trajectory[:, 1]   # Y_cam → Z_enu

# === Align origins ===
trajectory_ENU -= trajectory_ENU[0]
ground_truth -= ground_truth[0]

# === Plot both VO and Ground Truth ===
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory_ENU[:, 0], trajectory_ENU[:, 1], trajectory_ENU[:, 2], marker='o', label='ORB w/ Depth + PnP')
ax.plot(ground_truth[:len(trajectory_ENU), 0], ground_truth[:len(trajectory_ENU), 1], ground_truth[:len(trajectory_ENU), 2], label='Ground Truth', color='red')
ax.set_title("Monocular Visual Odometry vs Ground Truth")
ax.set_xlabel("X (East)")
ax.set_ylabel("Y (North)")
ax.set_zlabel("Z (Up)")
ax.legend()
plt.show()
