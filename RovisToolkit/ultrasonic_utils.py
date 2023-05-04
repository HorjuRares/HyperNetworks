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
import math
import cv2
import global_config

# TODO repair dependencies from removal of ultrasonic configs from global_config.py
CFG = global_config.cfg


def rot(x, y, theta):
    x_r = math.cos(theta) * x - math.sin(theta) * y
    y_r = math.sin(theta) * x + math.cos(theta) * y
    return x_r, y_r


def rot_t(pt, origin, theta):
    """
    Rotate by theta and translate with origin
    :return: x, y
    """
    x, y = rot(pt[0], pt[1], theta)
    x += origin[0]
    y += origin[1]
    return int(x), int(y)


def us2img(us_sequence, color=(0, 255, 0), size=(500, 500)):
    if len(us_sequence) != CFG.USONICS.COUNT_FRONT + CFG.USONICS.COUNT_BACK:
        return None
    rows = size[0]
    cols = size[1]
    vehicle_position = (rows // 2, cols // 2)
    vehicle_heading = math.radians(90)
    image = np.zeros((rows, cols, 3), dtype=np.uint8)

    # US maximum range is at 80% of the maximum available space
    scale_factor = 0.8 * (rows // 2) / CFG.USONICS.MAX_RANGE
    angle_increment_front = CFG.USONICS.FRONT_FOV / CFG.USONICS.COUNT_FRONT
    angle_increment_back = CFG.USONICS.REAR_FOV / CFG.USONICS.COUNT_BACK
    start_us_front_angle = CFG.USONICS.FRONT_FOV / 2
    start_us_back_angle = 3.14 + CFG.USONICS.REAR_FOV / 2

    us_info = list()
    for idx in range(CFG.USONICS.COUNT_FRONT):
        angle = start_us_front_angle - angle_increment_front * idx
        us_info.append((us_sequence[idx], angle))

    for idx in range(CFG.USONICS.COUNT_BACK):
        angle = start_us_back_angle - angle_increment_back * idx
        us_info.append((us_sequence[CFG.USONICS.COUNT_FRONT + idx], angle))

    idx = 0
    poly_points = list()

    for us in us_info:
        x1 = scale_factor * 0  # sensor translation
        y1 = scale_factor * 0  # sensor translation
        x2 = scale_factor * us[0] * math.cos(-us[1])
        y2 = scale_factor * us[0] * math.sin(-us[1])

        # Rotate with -vehicle_heading
        rot_x1, rot_y1 = rot(x1, y1, -vehicle_heading)
        rot_x2, rot_y2 = rot(x2, y2, -vehicle_heading)

        # Translate with vehicle_position
        us_start = (int(rot_x1 + vehicle_position[0]), int(rot_y1 + vehicle_position[1]))
        us_stop = (int(rot_x2 + vehicle_position[0]), int(rot_y2 + vehicle_position[1]))

        # cv2.line(image, us_start, us_stop, (0, 255, 0), 2)
        if idx == 0 or idx == CFG.USONICS.COUNT_FRONT:
            poly_points.append(us_start)
        poly_points.append(us_stop)
        if idx == CFG.USONICS.COUNT_FRONT - 1 or idx == CFG.USONICS.COUNT_FRONT + CFG.USONICS.COUNT_BACK - 1:
            poly_points.append(us_stop)
        idx += 1

    # Draw polygon
    if CFG.USONICS.COUNT_FRONT:
        poly_points_front = np.array(poly_points[:len(poly_points) // 2]).reshape((-1, 1, 2))
        image = cv2.fillPoly(image, [poly_points_front], color)
        poly_points_back = np.array(poly_points[len(poly_points) // 2:]).reshape((-1, 1, 2))
        image = cv2.fillPoly(image, [poly_points_back], color)
    else:
        poly_points_front = np.array(poly_points).reshape((-1, 1, 2))
        image = cv2.fillPoly(image, [poly_points_front], color)

    # Draw vehicle
    cv2.line(
        image,
        rot_t((0, -13), vehicle_position, -vehicle_heading),
        rot_t((33, 0), vehicle_position, -vehicle_heading),
        (255, 255, 255),
        3
    )
    cv2.line(
        image,  # img
        rot_t((0, 13), vehicle_position, -vehicle_heading),  # pt1
        rot_t((33, 0), vehicle_position, -vehicle_heading),  # pt2
        (255, 255, 255),  # color
        3  # thickness
    )
    cv2.line(
        image,  # img
        rot_t((0, -13), vehicle_position, -vehicle_heading),  # pt1
        rot_t((0, 13), vehicle_position, -vehicle_heading),  # pt2
        (255, 255, 255),  # color
        3  # thickness
    )

    cv2.putText(image, "US range: {}".format(CFG.USONICS.MAX_RANGE), (220, 30), cv2.FONT_HERSHEY_COMPLEX, 1.0,
                (255, 255, 255))
    cv2.putText(image, "US front: {}".format(CFG.USONICS.COUNT_FRONT), (220, 60), cv2.FONT_HERSHEY_COMPLEX, 1.0,
                (255, 255, 255))
    cv2.putText(image, "US rear: {}".format(CFG.USONICS.COUNT_BACK), (220, 90), cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255))

    return image
