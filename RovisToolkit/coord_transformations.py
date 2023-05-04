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
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

rot_x = 0
rot_y = 0
rot_z = 0
rot_delta = 0.1

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.25, bottom=0.29)
axcolor = 'lightgoldenrodyellow'
ax_x = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
ax_y = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
ax_z = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)

s_rot_x = Slider(ax_x, 'Rot X', 0, 2 * math.pi, valinit=rot_x, valstep=rot_delta)
s_rot_y = Slider(ax_y, 'Rot Y', 0, 2 * math.pi, valinit=rot_y, valstep=rot_delta)
s_rot_z = Slider(ax_z, 'Rot Z', 0, 2 * math.pi, valinit=rot_z, valstep=rot_delta)

from matplotlib.patches import FancyArrowPatch
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def rotate_x(xs, ys, zs, angle):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)]])
    r_x = []
    r_y = []
    r_z = []
    for x, y, z in zip(xs, ys, zs):
        pt_r = R_x @ np.array([x, y, z])
        r_x.append(pt_r[0])
        r_y.append(pt_r[1])
        r_z.append(pt_r[2])
    r_x = np.array(r_x)
    r_y = np.array(r_y)
    r_z = np.array(r_z)
    return r_x, r_y, r_z


def rotate_y(xs, ys, zs, angle):
    R_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)]])
    r_x = []
    r_y = []
    r_z = []
    for x, y, z in zip(xs, ys, zs):
        pt_r = R_y @ np.array([x, y, z])
        r_x.append(pt_r[0])
        r_y.append(pt_r[1])
        r_z.append(pt_r[2])
    r_x = np.array(r_x)
    r_y = np.array(r_y)
    r_z = np.array(r_z)
    return r_x, r_y, r_z


def rotate_z(xs, ys, zs, angle):
    R_z = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1]])
    r_x = []
    r_y = []
    r_z = []
    for x, y, z in zip(xs, ys, zs):
        pt_r = R_z @ np.array([x, y, z])
        r_x.append(pt_r[0])
        r_y.append(pt_r[1])
        r_z.append(pt_r[2])
    r_x = np.array(r_x)
    r_y = np.array(r_y)
    r_z = np.array(r_z)
    return r_x, r_y, r_z


def update(val):
    ax.clear()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    arrow_prop_dict = dict(mutation_scale=10, arrowstyle='->', shrinkA=0, shrinkB=0)
    a = Arrow3D([0, 1], [0, 0], [0, 0], **arrow_prop_dict, color='r')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 1], [0, 0], **arrow_prop_dict, color='b')
    ax.add_artist(a)
    a = Arrow3D([0, 0], [0, 0], [0, 1], **arrow_prop_dict, color='g')
    ax.add_artist(a)

    #r_x, r_y, r_z = rotate_x(x, y, z, s_rot_x.val)
    #r_x, r_y, r_z = rotate_y(r_x, r_y, r_z, s_rot_y.val)
    #r_x, r_y, r_z = rotate_z(r_x, r_y, r_z, s_rot_z.val)
    #ax.plot(r_x, r_y, r_z)
    ax.margins(x=0)
    fig.canvas.draw_idle()


s_rot_x.on_changed(update)
s_rot_y.on_changed(update)
s_rot_z.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    s_rot_x.reset()
    s_rot_y.reset()
    s_rot_z.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('World', 'Vehicle', 'Camera'), active=0)


def world2vehicle(w_xs, w_ys, w_zs):
    pass


def colorfunc(label):
    pass
    # l.set_color(label)
    # fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

update(0)
plt.show()
