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
 * image_utils.py
 *
 *  Created on: 29.08.2021
 *      Author: Sorin Grigorescu
"""

import numpy as np
import torch
import os
import cv2
from RovisToolkit.data_transformations import *

def nhwc_2_nchw(batch):
    """ Transposes a batch of images to NCHW representation """
    if (batch.shape[1] > 3 or batch.shape[2] > 3) and batch.shape[3] <= 3:
        if isinstance(batch, torch.Tensor):
            return batch.permute(0, 3, 1, 2)
        else:
            return np.transpose(batch, (0, 3, 1, 2))
    else:
        return batch

def nchw_2_nhwc(batch):
    """ Transposes a batch of images to NHWC representation """
    if len(batch.shape) == 3:
        if (batch.shape[0] <= 3 or batch.shape[1] <= 3) and batch.shape[2] > 3:
            if isinstance(batch, torch.Tensor):
                return batch.permute(1, 2, 0)
            else:
                return np.transpose(batch, (1, 2, 0))
        else:
            return batch
    else:
        if (batch.shape[1] <= 3 or batch.shape[2] <= 3) and batch.shape[3] > 3:
            if isinstance(batch, torch.Tensor):
                return batch.permute(0, 2, 3, 1)
            else:
                return np.transpose(batch, (0, 2, 3, 1))
        else:
            return batch

def imagebatch_2_tensor(batch, transforms, img_input_shape=None):
    """ Converts a batch of images into a NCHW tensor
        e.g. in/out shape: (2, 3, 640, 640)"""
    assert batch is not None

    # Apply transformation on the raw batch of input images
    batch = transform_imagesbatch(batch, transforms)

    if img_input_shape:
        hw_shape = img_input_shape[1:3]  # Get the HW from the input shape
        wh_shape = hw_shape[::-1]  # Convert HW representation to WH for resizing
        batch = [cv2.resize(im, dsize=tuple(wh_shape), interpolation=cv2.INTER_NEAREST) for im in batch]

    if isinstance(batch, torch.Tensor):
        batch = batch.cpu().float().numpy()

    img_numpy = np.array(batch, dtype=np.float32)

    # Normalise images
    if np.max(img_numpy) > 1.:
        img_numpy /= 255.

    # Transpose channels to NCHW representation
    img_numpy = nhwc_2_nchw(img_numpy)

    img_numpy = np.ascontiguousarray(img_numpy)
    batch_tensor = torch.tensor(img_numpy, dtype=torch.float32)

    # Apply transformation on the torch tensor batch
    batch_tensor = transform_tensorbatch(batch_tensor, transforms)

    return batch_tensor

def imagefolder_2_batches(imgs_folder, batch_size, img_input_shape=None, sample_limit=-1):
    """ Returns a list of image batches from a folder containing images. """
    imgs_paths = os.listdir(imgs_folder)
    imgs_paths = [os.path.join(imgs_folder, im) for im in imgs_paths if
                  "png" in im.lower() or "jpg" in im.lower() or "jpeg" in im.lower()]

    # Parse the input images folder
    batches = list()
    for idx in range(0, len(imgs_paths), batch_size):
        if sample_limit != -1 and idx == sample_limit:
            break

        # Create an input batch
        batch = list()
        for k in range(batch_size):
            if idx + k < len(imgs_paths):
                img = cv2.imread(imgs_paths[idx + k])
                if img_input_shape:
                    hw_shape = img_input_shape[1:3]  # Get the HW from the input shape
                    wh_shape = hw_shape[::-1]  # Convert HW representation to WH for resizing
                    img = cv2.resize(img, tuple(wh_shape), interpolation=cv2.INTER_NEAREST)
                img = np.asarray(img, dtype=np.float32)
                batch.append(img)
        batches.append(batch)
    return batches

def decode_semseg(semseg_output, colormap):
    if len(semseg_output.shape) > 2:
        # If we have output from scene segmentation network (argmax needed)
        semseg = np.argmax(semseg_output, axis=0)
    else:
        # If we have output from dataset (argmax not needed)
        semseg = semseg_output

    semseg = np.asarray(semseg, dtype=np.float32)

    decoded = np.zeros((semseg.shape[0], semseg.shape[1], 3), dtype=np.float32)
    for i in range(semseg.shape[0]):
        for j in range(semseg.shape[1]):
            decoded[i, j] = colormap[int(semseg[i, j])] # self.CGNet_hyp['colormap'][int(semseg[i, j])]
    decoded /= 255.0

    return decoded


def unnormalize_detections(normalized_detections, scale_width, scale_height):
    unnormalized_detections = list()
    for bbox in normalized_detections:
        if bbox.shape[0] == 0:
            continue
        bbox[:, [0, 2]] *= scale_width
        bbox[:, [1, 3]] *= scale_height
        unnormalized_detections.append(bbox)
    return unnormalized_detections


def decode_depth(rgb_encoded_depth, scale=100000.):
    depth = np.zeros(rgb_encoded_depth.shape[:2], dtype=np.int32)
    for i in range(rgb_encoded_depth.shape[0]):
        for j in range(rgb_encoded_depth.shape[1]):
            px = rgb_encoded_depth[i, j]
            for k in range(rgb_encoded_depth.shape[2]):
                depth[i, j] |= int(px[k]) << (8 * k)

    depth = depth / scale
    return depth


def image2depth(in_image: np.ndarray = None, scale: float = 16.0):
    # see depth2image
    depth = np.zeros(shape=(in_image.shape[0], in_image.shape[1]), dtype=np.int32)

    for k in range(in_image.shape[2]):
        depth[:, :] |= (np.int32(in_image[:, :, k]) << (8 * k))

    return np.float32(np.float32(depth) / scale)
