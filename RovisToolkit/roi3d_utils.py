"""
Copyright (c) RovisLab
RovisDojo: RovisLab neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (s.grigorescu@unitbv.ro)
"""

from math import cos, sin
from typing import List, Tuple
from pyquaternion import Quaternion
from matplotlib.axes import Axes
import pyqtgraph.opengl as gl
import numpy as np
import cv2


def view_points(points: np.ndarray, view: np.ndarray, normalize: bool) -> np.ndarray:
    """
    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.
    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
        all zeros) and normalize=False
    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]

    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    return points


class Box:
    """ Simple data class representing a 3d box including, center, size, rotation and class label """

    def __init__(self,
                 frame_index: int,
                 roi_index: int,
                 center: List[float],
                 size: List[float],
                 orientation: List[any],
                 name: str = None,
                 color: tuple = (1., 1., 1., 1.)):
        """
        :param center: Center of box given as x, y, z.
        :param size: Size of box in width, length, height.
        :param orientation: Box orientation.
        :param name: Box name, optional. Can be used e.g. for denote category name.
        """
        assert not np.any(np.isnan(center))
        assert not np.any(np.isnan(size))
        assert len(center) == 3
        assert len(size) == 3
        assert frame_index is not None
        assert roi_index is not None

        self.frame_index = frame_index
        self.roi_index = roi_index
        self.center = np.array(center)
        self.wlh = np.array(size)
        self.__to_quaternion(orientation)
        self.name = name
        self.color = color

    def __eq__(self, other):
        """
        Comparable method to other object of type Box.
        """
        center = np.allclose(self.center, other.center)
        wlh = np.allclose(self.wlh, other.wlh)
        orientation = np.allclose(self.orientation.elements, other.orientation.elements)

        return center and wlh and orientation

    def __repr__(self):
        repr_str = 'xyz: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'wlh: [{:.2f}, {:.2f}, {:.2f}], ' \
                   'rot axis: [{:.2f}, {:.2f}, {:.2f}]' \
                   'name: {}'

        return repr_str.format(self.center[0], self.center[1], self.center[2],
                               self.wlh[0], self.wlh[1], self.wlh[2],
                               self.orientation.axis[0], self.orientation.axis[1], self.orientation.axis[2],
                               self.name)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """
        Return a rotation matrix.
        :return: <np.float: 3, 3>. The box's rotation matrix.
        """
        return self.orientation.rotation_matrix

    @property
    def get_frame_roi_indexes(self) -> Tuple:
        """
        Return the frame index.
        :return: int. The frame index where this box is detected.
        """
        return self.frame_index, self.roi_index

    def set_color(self, color):
        """
        Sets the color for the bounding box.
        :param color: Tuple(float:4): R G B A values, normalized in between [0.0, 1.0]
        """
        self.color = color

    def __to_quaternion(self, orientation: List[float]):
        """
        Gets the 3 angles roll, pitch, yaw and generates a quaternion
        :param orientation:<List[float]>. Get quaternion related to these 3 angles
        """
        roll, pitch, yaw = [float(angle_rad) for angle_rad in orientation]

        w = cos(roll * 0.5) * cos(pitch * 0.5) * cos(yaw * 0.5) + sin(roll * 0.5) * sin(pitch * 0.5) * sin(yaw * 0.5)
        x = sin(roll * 0.5) * cos(pitch * 0.5) * cos(yaw * 0.5) - cos(roll * 0.5) * sin(pitch * 0.5) * sin(yaw * 0.5)
        y = cos(roll * 0.5) * sin(pitch * 0.5) * cos(yaw * 0.5) + sin(roll * 0.5) * cos(pitch * 0.5) * sin(yaw * 0.5)
        z = cos(roll * 0.5) * cos(pitch * 0.5) * sin(yaw * 0.5) - sin(roll * 0.5) * sin(pitch * 0.5) * cos(yaw * 0.5)

        self.orientation = Quaternion(w, x, y, z)

    def translate(self, x: np.ndarray) -> None:
        """
        Applies a translation.
        :param x: <np.float: 3, 1>. Translation in x, y, z direction.
        """
        self.center += x

    def rotate(self, quaternion: Quaternion) -> None:
        """
        Rotates box.
        :param quaternion: Rotation to apply.
        """
        self.center = np.dot(quaternion.rotation_matrix, self.center)
        self.orientation = quaternion * self.orientation

    def corners(self, wlh_factor: float = 1.0) -> np.ndarray:
        """
        Returns the bounding box corners.
        :param wlh_factor: Multiply w, l, h by a factor to scale the box.
        :return: <np.float: 3, 8>. First four corners are the ones facing forward.
            The last four are the ones facing backwards.
        """
        w, h, l = self.wlh * wlh_factor

        # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
        x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
        y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
        z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
        corners = np.vstack((x_corners, y_corners, z_corners))

        # Rotate
        corners = np.dot(self.orientation.rotation_matrix, corners)

        # Translate
        x, y, z = self.center
        corners[0, :] = corners[0, :] + x
        corners[1, :] = corners[1, :] + y
        corners[2, :] = corners[2, :] + z

        return corners

    def bottom_corners(self) -> np.ndarray:
        """
        Returns the four bottom corners.
        :return: <np.float: 3, 4>. Bottom corners. First two face forward, last two face backwards.
        """
        return self.corners()[2, 3, 7, 6]

    def top_corners(self) -> np.ndarray:
        """
        Returns the four top corners.
        :return: <np.float: 3, 4>. Top corners. First two face forward, last two face backwards.
        """
        return self.corners()[0, 1, 5, 4]

    def front_face(self) -> np.ndarray:
        """
        Returns the front face corners.
        :return: <np.float: 3, 4>. Returns front face.
        """
        return self.corners()[0:4]

    def back_face(self) -> np.ndarray:
        """
        Returns the front face corners.
        :return: <np.float: 3, 4>. Returns front face.
        """
        return self.corners()[4:8]

    def get_mesh_box(self, edge_color: Tuple = ('r', 'g', 'b', 'a')):
        """
        Instantiates a GL_MeshObject representing the bounding box.
        :param edge_color: (Tuple<4>) R G B A values, 1 = 255, 0 = 0 To represent the color of the line delimiting triangles
        :param line_width: Mesh line width
        """
        if edge_color is None:
            edge_color = (0, 0, 0, 1)

        vertices = np.stack(self.corners(), axis=-1)
        faces = np.array([[0, 3, 2], [0, 1, 2],
                          [0, 4, 5], [0, 1, 5],
                          [0, 4, 7], [0, 3, 7],
                          [5, 4, 7], [5, 6, 7],
                          [5, 6, 2], [5, 1, 2],
                          [3, 2, 6], [3, 7, 6]])
        colors = [[self.color for i in range(12)]]
        cube = gl.GLMeshItem(vertexes=vertices, faces=faces, faceColors=colors, drawEdges=True, edgeColor=edge_color,
                             computeNormals=False)

        return cube

    def get_wireframe_box(self, line_width: int = 1):
        """
        Renders GL_Lines for each vertex connection resulting a wire-frame bounding box.
        :param line_width: Line width, 1 being default.
        :return: Set of GL_Lines composing the bounding box.
        """
        def extract_vertices(V, E) -> np.array:
            """
            Given the edges, it groups the vertices by consecutive pairs.
            :param V: <np.array(8, 3)>. Bounding box corners
            :param E: <np.array(14, 2)>. Bounding box edges, as seen in OpenGL X o Y o Z
            :return: <np.array(28, 3)>. Array of vertices in consecutive order for GLLinePlotter
            """
            return np.array([[V[pos[0]],
                              V[pos[1]]] for pos in E]).flatten().reshape((2*len(E), 3))

        vertices = np.stack(self.corners(), axis=-1)
        connexions = [
            [0, 2], [1, 3],  # Heading (optional)
            [0, 1], [1, 2], [2, 3], [3, 0],  # Front
            [5, 4], [4, 7], [7, 6], [6, 5],  # Back
            [1, 5], [2, 6], [0, 4], [3, 7],  # Sides
        ]

        vertices_for_plotting = extract_vertices(vertices, connexions)
        lines_set = gl.GLLinePlotItem(pos=vertices_for_plotting, width=line_width, color=self.color, mode='lines')
        return lines_set

    def render(self,
               axis: Axes,
               view: np.ndarray = np.eye(3),
               normalize: bool = False,
               linewidth: float = 2) -> None:
        """
        Renders the box in the provided Matplotlib axis.
        :param axis: Axis onto which the box should be drawn.
        :param view: <np.array: 3, 3>. Define a projection in needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param linewidth: Width in pixel of the box sides.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                axis.plot([prev[0], corner[0]], [prev[1], corner[1]], color=color, linewidth=linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            axis.plot([corners.T[i][0], corners.T[i + 4][0]],
                      [corners.T[i][1], corners.T[i + 4][1]],
                      color=self.color, linewidth=linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], self.color)
        draw_rect(corners.T[4:], self.color)

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        axis.plot([center_bottom[0], center_bottom_forward[0]],
                  [center_bottom[1], center_bottom_forward[1]],
                  color=self.color, linewidth=linewidth)

    def get_rendering_corners(self, view: np.ndarray = np.eye(3), normalize: bool = False):
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]
        return corners

    def render_cv2(self,
                   im: np.ndarray,
                   view: np.ndarray = np.eye(3),
                   normalize: bool = False,
                   colors: Tuple = ((0, 0, 255), (255, 0, 0), (155, 155, 155)),
                   linewidth: int = 2) -> None:
        """
        Renders box using OpenCV2.
        :param im: <np.array: width, height, 3>. Image array. Channels are in BGR order.
        :param view: <np.array: 3, 3>. Define a projection if needed (e.g. for drawing projection in an image).
        :param normalize: Whether to normalize the remaining coordinate.
        :param colors: ((R, G, B), (R, G, B), (R, G, B)). Colors for front, side & rear.
        :param linewidth: Linewidth for plot.
        """
        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]

        def draw_rect(selected_corners, color):
            prev = selected_corners[-1]
            for corner in selected_corners:
                cv2.line(im,
                         (int(prev[0]), int(prev[1])),
                         (int(corner[0]), int(corner[1])),
                         color, linewidth)
                prev = corner

        # Draw the sides
        for i in range(4):
            cv2.line(im,
                     (int(corners.T[i][0]), int(corners.T[i][1])),
                     (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                     colors[2][::-1], linewidth)

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners.T[:4], colors[0][::-1])
        draw_rect(corners.T[4:], colors[1][::-1])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        cv2.line(im,
                 (int(center_bottom[0]), int(center_bottom[1])),
                 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                 colors[0][::-1], linewidth)

    @staticmethod
    def euler_to_quaternion(yaw, pitch, roll):
        """Method to transform from euler angles to an quaternion object.
        @param: roll: <float> Roll angle around OX.
        @param: pitch: <float> Pitch angle around OY.
        @param: yaw: <float> Yar angle around OZ.
        """
        cy = cos(yaw / 2)
        sy = sin(yaw / 2)
        cp = cos(pitch / 2)
        sp = sin(pitch / 2)
        cr = cos(roll / 2)
        sr = sin(roll / 2)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return [qw, -qx, qy, -qz]

    def rotate_euler(self, rotation_angles):
        """Method to rotate box object around provided Euler angles..
        @param: roll: <float> Roll angle around OX.
        @param: pitch: <float> Pitch angle around OY.
        @param: yaw: <float> Yar angle around OZ.
        """
        yaw, pitch, roll = [float(angle) for angle in rotation_angles]
        qw, qx, qy, qz = self.euler_to_quaternion(yaw, pitch, roll)
        rotation = Quaternion(qw, qx, qy, qz).inverse
        self.rotate(rotation)




