"""
Copyright (c) RovisLab
RovisDojo: RovisLab neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (s.grigorescu@unitbv.ro)
"""

"""
Reader interface for .conf files assigned to each sensor existing in RovisVision project.
===============================================================
1. Nuscenes Cameras

                        CAM_FRONT
                        (tr, rot)
        CAM_FRONT_LEFT              CAM_FRONT RIGHT
          (tr,rot)                      (tr,rot)

        CAM_BACK_LEFT               CAM_BACK_RIGHT
          (tr,rot)                      (tr,rot)
                        CAM_BACK
                        (tr,rot)

    *** Read Translation and Rotation for each camera.

2. Kitti Cameras
  1.65m     CAM0,1(gray) <-0.6m-> CAM2,3(color)
    ^               Velodyne HDL-64E
    |               2 Grayscale Cameras
    |               2 Color Cameras
    |
    |
    |
    |
  GROUND

   *** Read calibration parameters for each camera.
===============================================================
"""

from absl import app, flags, logging
import os
import io, libconf
import numpy as np
import math
from RovisToolkit import coordinate_frames_utils as frames

from global_config import *
CFG = cfg


class PinholeCameraSensorModel(object):
    def __init__(self, calibration_file):
        assert os.path.exists(calibration_file), 'Path to calibration file invalid.'
        self.camera_type = None
        self.rotation = np.zeros((3))
        self.translation = np.zeros((3))
        self.width_ = -1
        self.height_ = -1
        self.channels_ = -1
        self.fx_px_ = -1
        self.fy_px_ = -1
        self.cx_ = -1
        self.cy_ = -1
        self.sx_ = -1
        self.sy_ = -1
        self.fx_m_ = -1
        self.fy_m_ = -1
        self.d_ = []
        self.K_ = []
        # self.parse_rovis_vision_pipeline_file(calibration_file) # Deprecated; used for reading the old Frame variable
        self.read_rovis_vision_calib_file(calibration_file)

        # Transform points from vehicle coordinates to camera coordinates
        self.M_cam2veh = frames.transform_mat_XYZ(self.translation[0], self.translation[1], self.translation[2],
              np.deg2rad(self.rotation[0]), np.deg2rad(self.rotation[1]), np.deg2rad(self.rotation[2]))
        self.M_cam2veh = np.vstack([self.M_cam2veh, [0., 0., 0., 1.]])

        # Calculate camera to vehicle transformation matrix (inverse of M_veh2cam)
        self.M_veh2cam = np.linalg.inv(self.M_cam2veh)

    def __str__(self):
        str_display = "Width:\t\t{0}px\nHeight:\t\t{1}px\nChannels:\t{2}\n".format(self.width_, self.height_,
                                                                                    self.channels_)
        str_display += "Focal length x:\t\t{0}px\nFocal length y:\t\t{1}px\n".format(self.fx_px_, self.fy_px_)
        str_display += "Optical center x:\t{0}px\nOptical center y:\t{1}px\n".format(self.cx_, self.cy_)
        str_display += "Rotation:\t\t{0}\n".format(self.rotation)
        str_display += "Translation:\t{0}\n".format(self.translation)
        return str_display

    def parse_rovis_vision_pipeline_file(self, rovis_vision_pipeline_file):
        """Open conf file, read and parse contents."""
        with io.open(rovis_vision_pipeline_file) as f:
            sections = libconf.load(f)
            self.parse_rovis_vision_filters_section(sections['Filters'])

    def parse_rovis_vision_filters_section(self, filters):
        """Given the filters config parameters, parse for each camera."""
        bFoundCam = False
        for entry in filters:
            cam_id = filters[entry]['ID']
            if self.camera_id == cam_id:
                camera_info = filters[entry]
                calib_file_path = camera_info['Parameters'][0]['value']
                self.read_rovis_vision_calib_file(calib_file_path)
                self.camera_type = camera_info['Type']
                self.set_camera_pose(camera_info['Frame'])
                bFoundCam = True
        assert bFoundCam, 'Camera ID not found in RovisVision pipeline configuration file.'

    def set_camera_pose(self, rovis_rot_tr_str):
        """Set camera translation and rotation"""
        assert type(rovis_rot_tr_str) == str, 'Frame parameter from camera, is not parsed correctly.'
        rot_tr_values = rovis_rot_tr_str.split(',')
        self.translation = [float(val) for val in rot_tr_values[0:3]]
        self.rotation = [float(val) for val in rot_tr_values[3:6]]

    def read_rovis_vision_calib_file(self, calib_file_path):
        """Read configuration file and store the content in a dict."""
        assert os.path.exists(calib_file_path), 'Path to camera calibration file invalid.'
        with io.open(calib_file_path) as f:
            sections = libconf.load(f)

            # Read extrinsics
            self.rotation[0] = sections['Pose']['Rotation']['x']
            self.rotation[1] = sections['Pose']['Rotation']['y']
            self.rotation[2] = sections['Pose']['Rotation']['z']
            self.translation[0] = sections['Pose']['Translation']['x']
            self.translation[1] = sections['Pose']['Translation']['y']
            self.translation[2] = sections['Pose']['Translation']['z']

            # Read intrinsics
            try:
                self.width_ = int(sections['image_width'])
                self.height_ = int(sections['image_height'])
                self.channels_ = sections['LeftSensor']['channels']
                self.fx_px_ = sections['LeftSensor']['focal_length_x']
                self.fy_px_ = sections['LeftSensor']['focal_length_y']
                self.cx_ = sections['LeftSensor']['optical_center_x']
                self.cy_ = sections['LeftSensor']['optical_center_y']
                self.sx_ = sections['LeftSensor']['pixel_size_x']
                self.sy_ = sections['LeftSensor']['pixel_size_y']
            except KeyError:
                print("Invalid camera calibration file: {}!".format(calib_file_path))
                exit(-1)

            # Calculate focal length in meters (focal length in pixels * pixel size)
            self.fx_m_ = self.fx_px_ * self.sx_
            self.fy_m_ = self.fy_px_ * self.sy_

            self.d_ = np.array([sections['LeftSensor']['dist_coeff_0'],
                                sections['LeftSensor']['dist_coeff_1'],
                                sections['LeftSensor']['dist_coeff_2'],
                                sections['LeftSensor']['dist_coeff_3'],
                                sections['LeftSensor']['dist_coeff_4']])

            self.K_ = np.array([[self.fx_px_, 0, self.cx_],
                                [0, self.fy_px_, self.cy_],
                                [0, 0, 1]])

    def world2sensor(self, world_pts):
        """Converts an array of 3D points to 2D image coordinates"""
        image_pts = list()
        for xyz in world_pts:
            if xyz[2] > 0:
                x = self.fx_px_ * (xyz[0] / xyz[2]) + self.cx_
                y = self.fy_px_ * (xyz[1] / xyz[2]) + self.cy_
                image_pts.append([x, y])
        return np.array(image_pts)

    def set_cam_id(self, cam_id):
        """Assigns camera id"""
        self.camera_id = cam_id

    def set_type(self, camera_type: str):
        """Sets camera type"""
        self.camera_type = camera_type

    def set_rot_tr(self, rot_tr_info):
        """Set camera translation and rotation"""
        assert type(rot_tr_info) == str, 'Frame parameter from camera, is not parsed correctly.'
        rot_tr_values = rot_tr_info.split(',')
        self.translation = [float(val) for val in rot_tr_values[0:3]]
        self.rotation = [float(val) for val in rot_tr_values[3:6]]

    def get_image_size(self):
        """Return width and height of the image."""
        return self.height_, self.width_

    def get_proj_matrix(self):
        return np.matmul(self.K_, np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0]]))

    def get_intrinsic_matrix(self):
        """Return Intrinsic parameters matrix."""
        return self.K_

    def get_rotation(self):
        """Get camera rotation in euler angles."""
        return self.rotation

    def get_translation(self):
        """Get camera translation."""
        return np.array(self.translation)


def tu_sensor_model(_argv):
    np.set_printoptions(suppress=True)
    # M_veh2cam = frames.transform_mat_XYZ(1.51, 0.015, 1.70, np.deg2rad(0.), np.deg2rad(90.), np.deg2rad(-90.))
    # print(M_veh2cam)

    calibration_file = r"C:\dev\src\RovisLab\RovisVision\etc\calibration\nuscenes\nuscenes_camera_front.cal"
    sensor_model = PinholeCameraSensorModel(calibration_file)
    print(sensor_model)

if __name__ == '__main__':
    try:
        app.run(tu_sensor_model)
    except SystemExit:
        pass
