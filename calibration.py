
import numpy as np
from datetime import datetime

def load_calibration(file_path):
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
    baseline = np.abs(P_right[0, 3] - P_left[0, 3]) / fx
    return K_left, fx, baseline

def load_timestamps(path):
    with open(path) as f:
        lines = f.readlines()
    timestamps = []
    for ts in lines:
        main_time, nano = ts[:19], ts[20:]
        dt = datetime.strptime(main_time, "%Y-%m-%d %H:%M:%S")
        sec = (dt - datetime(1970, 1, 1)).total_seconds() + float("0." + nano)
        timestamps.append(sec)
    timestamps = np.array(timestamps) - timestamps[0]
    return timestamps

