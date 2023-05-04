"""
Copyright (c) RovisLab
RovisDojo: RovisLab neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (s.grigorescu@unitbv.ro)
"""

import numpy as np
from absl import app
from RovisToolkit import pinhole_camera_sensor_model as pinhole

def generate_lane(theta_0, theta_1, theta_2, theta_3=0):
    x = np.linspace(0, 60, num=60)
    lane_pts_3d = []
    for i in range(len(x)):
        y = theta_0 + theta_1 * x[i] + theta_2 * x[i] ** 2 + theta_3 * x[i] ** 3
        z = 0
        w = 1
        lane_pts_3d.append([x[i], y, z, w])

    return np.array(lane_pts_3d)


def lane_3d_from_2d(camera_model, x, y):
    if (y - camera_model.cy_) == 0:
        return None

    Y = camera_model.translation[2]
    Z = (camera_model.fy_px_ * Y) / (y - camera_model.cy_)
    X = (Z * (x - camera_model.cx_)) / camera_model.fx_px_

    return np.array([X, Y, Z, 1])


def tu_LaneUtils(_argv):
    calibration_file = r'../etc/calibration/carla/carla_front.cal'
    cam = pinhole.PinholeCameraSensorModel(calibration_file)
    print(cam)
    print(lane_3d_from_2d(cam, 340, 50))

if __name__ == '__main__':
    try:
        app.run(tu_LaneUtils)
    except SystemExit:
        pass