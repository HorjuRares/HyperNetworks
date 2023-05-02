"""
Copyright (c) RovisLab
RovisDojo: RovisLab neural network design, training and annotation

All rights reserved. This program and the accompanying materials
are made available under the terms of the Proprietary license
which accompanies this distribution.

For any commercial applications, details and software licensing, 
please contact Prof. Sorin Grigorescu (s.grigorescu@unitbv.ro)
"""

import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, inplanes, outplanes, stride):
        super(Downsample, self).__init__()
        self.conv1x1 = nn.Conv2d(inplanes, outplanes, 1, stride, bias=False)
        self.norm = nn.BatchNorm2d(outplanes)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.norm(x)
        return x

def nostride_dilate(layer, dilate):
    for m in layer.modules():
        # the convolution with stride
        if isinstance(m, nn.Conv2d):
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate // 2, dilate // 2)
                    m.padding = (dilate // 2, dilate // 2)
            # other convolutions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
