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
from absl import app, flags, logging
import math as m
from scipy.spatial.transform import Rotation as Rot

PI = 3.14159265358979
DEG2RAD = (PI / 180.)        # degree to radian
RAD2DEG = (180. / PI)        # radian to degree

def transform_mat_XYZ(trans_x, trans_y, trans_z, rot_x, rot_y, rot_z):
    rot = euler2mat_XYZ(rot_x, rot_y, rot_z)
    return np.hstack((rot, [[trans_x], [trans_y], [trans_z]]))

def euler2mat_XYZ(x, y, z):
    """
    Returns matrix for rotations around z, y and x axes.
    Uses the convention of static-frame rotation around the z, then y, then x axis.
    Reference: https://github.com/matthew-brett/transforms3d/blob/master/transforms3d/taitbryan.py

    :param z: Rotation angle in radians around z-axis (performed first)
    :param y: Rotation angle in radians around y-axis
    :param x: Rotation angle in radians around x-axis (performed last)
    :return:  Rotation matrix giving same rotation as for given angles, array shape (3,3)
    """
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(x), -np.sin(x)],
                      [0, np.sin(x), np.cos(x)]])

    rot_y = np.array([[np.cos(y), 0, np.sin(y)],
                      [0, 1, 0],
                      [-np.sin(y), 0, np.cos(y)]])

    rot_z = np.array([[np.cos(z), -np.sin(z), 0],
                      [np.sin(z), np.cos(z), 0],
                      [0, 0, 1]])

    # R = Rz * Ry * Rx
    return rot_z @ rot_y @ rot_x

def R2euler(R):
    """
    :param R: 3x3 rotation matrix
    :return:  euler angles, where y_pitch will always be positive
    """
    x_roll = m.atan2(R[2, 1], R[2, 2])
    y_pitch = m.atan2(-R[2, 0], m.sqrt(R[2, 1]*R[2, 1] + R[2, 2]*R[2, 2]))
    z_yaw = m.atan2(R[1, 0], R[0, 0])
    return np.array([x_roll, y_pitch, z_yaw])

def R2quaternion(R):
    w = m.sqrt(max(0., 1 + R[0][0] + R[1][1] + R[2][2])) / 2.0
    x = m.sqrt(max(0., 1 + R[0][0] - R[1][1] - R[2][2])) / 2.0
    y = m.sqrt(max(0., 1 - R[0][0] + R[1][1] - R[2][2])) / 2.0
    z = m.sqrt(max(0., 1 - R[0][0] - R[1][1] + R[2][2])) / 2.0

    if (R[2][1] - R[1][2]) >= 0:
        x = abs(x)
    else:
        x = -abs(x)

    if (R[0][2] - R[2][0]) >= 0:
        y = abs(y)
    else:
        y = -abs(y)

    if (R[1][0] - R[0][1]) >= 0:
        z = abs(z)
    else:
        z = -abs(z)

    return np.array([x, y, z, w])

def quaternion2R(q):
    R = np.zeros((3, 3), dtype=float)

    X = q[0]
    Y = q[1]
    Z = q[2]
    W = q[3]

    xx = X * X
    xy = X * Y
    xz = X * Z
    xw = X * W

    yy = Y * Y
    yz = Y * Z
    yw = Y * W

    zz = Z * Z
    zw = Z * W

    R[0, 0] = 1 - 2 * (yy + zz)
    R[0, 1] = 2 * (xy - zw)
    R[0, 2] = 2 * (xz + yw)

    R[1, 0] = 2 * (xy + zw)
    R[1, 1] = 1 - 2 * (xx + zz)
    R[1, 2] = 2 * (yz - xw)

    R[2, 0] = 2 * (xz - yw)
    R[2, 1] = 2 * (yz + xw)
    R[2, 2] = 1 - 2 * (xx + yy)

    return R

def tu_Frames(_argv):
    eulers = list()
    eulers.append([2, 85, 20])      # 0
    #eulers.append([98, 85, 20])     # 1
    #eulers.append([2, 85, 100])     # 2
    #eulers.append([98, 85, 100])    # 3
    #eulers.append([2, 95, 20])      # 4
    #eulers.append([98, 95, 20])     # 5
    #eulers.append([2, 95, 100])     # 6
    #eulers.append([98, 95, 100])    # 7

    idx = 0
    for euler in eulers:
        print(idx, ":")
        idx = idx + 1
        x_euler = euler[0]
        y_euler = euler[1]
        z_euler = euler[2]

        # Test euler2rotmat using scipy
        R_scipy = Rot.from_euler('xyz', [x_euler, y_euler, z_euler], degrees=True)
        print(R_scipy.as_matrix(), "\n")

        # Own function
        R = euler2mat_XYZ(x_euler * DEG2RAD, y_euler * DEG2RAD, z_euler * DEG2RAD)
        #print(R, "\n")

        # Test rotmat2euler using scipy
        euler_scipy = R_scipy.as_euler('xyz', degrees=True)
        #print(euler_scipy, "\n")

        # Own function
        euler = R2euler(R)
        #print(euler * RAD2DEG, "\n")

        # Quaternions
        q_scipy = R_scipy.as_quat()
        #print(q_scipy, "\n")

        # Own function
        q = R2quaternion(R)
        #print(q, "\n")

        # Quaternion to rotation matrix
        R_scipy = Rot.from_quat(q_scipy)
        #print(R_scipy.as_matrix(), "\n")

        # Own function
        R = quaternion2R(q)
        print(R, "\n")

if __name__ == '__main__':
    try:
        app.run(tu_Frames)
    except SystemExit:
        pass
