
import os
import numpy as np
from pyproj import Proj

def load_oxts_ground_truth(oxts_folder):
    files = sorted([os.path.join(oxts_folder, f) for f in os.listdir(oxts_folder) if f.endswith(".txt")])
    
    lat_lon_alt = []
    for file in files:
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
    return np.vstack((x, y, z)).T

def convert_to_ENU(trajectory_cam):
    enu = np.zeros_like(trajectory_cam)
    enu[:, 0] = trajectory_cam[:, 2]
    enu[:, 1] = -trajectory_cam[:, 0]
    enu[:, 2] = -trajectory_cam[:, 1]
    return enu - enu[0]
