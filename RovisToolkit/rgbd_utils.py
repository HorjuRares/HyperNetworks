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
import os
import cv2
import torch
from RovisToolkit.image_utils import imagebatch_2_tensor, nhwc_2_nchw
from RovisToolkit.image_utils import decode_depth


def rgbdbatch_2_tensor(batch, device, input_data_transforms=[], img_input_shape=None):
    """ Converts a batch of RGBD images into a list of tuples, where each tuple contains an RGB and a Depth tensor"""
    # assert batch is not None

    # Convert the RGB images to tensors
    rgb_batch = list()
    for k in range(len(batch)):
        rgb_batch.append(batch[k][0])
    rgb_batch_tensor = imagebatch_2_tensor(rgb_batch, device, input_data_transforms, img_input_shape)

    # Convert the Depth images to tensors
    depth_batch = list()
    for k in range(len(batch)):
        depth_batch.append(batch[k][1])
    depth_batch = np.array(depth_batch, dtype=np.float32)
    depth_batch = np.ascontiguousarray(depth_batch)
    depth_batch_tensor = torch.tensor(depth_batch, dtype=torch.float32).to(device, non_blocking=True)

    # Merge tensors into list
    batch_tensor = list((rgb_batch_tensor, depth_batch_tensor))

    return batch_tensor


def rgbdfolder_2_batches(imgs_folder, batch_size, img_input_shape=None, sample_limit=-1):
    """ Returns a list of RGBD batches from a folder containing RGB and depth images. """
    rgb_paths = os.listdir(imgs_folder + '/left')
    rgb_paths = [os.path.join(imgs_folder + '/left', im) for im in rgb_paths if
                 "png" in im.lower() or "jpg" in im.lower() or "jpeg" in im.lower()]

    depth_paths = os.listdir(imgs_folder + '/right')
    depth_paths = [os.path.join(imgs_folder + '/right', im) for im in depth_paths if
                   "png" in im.lower() or "jpg" in im.lower() or "jpeg" in im.lower()]

    assert len(rgb_paths) == len(depth_paths)

    # Parse the input images folder
    batches = list()
    for idx in range(0, len(rgb_paths), batch_size):
        if sample_limit != -1 and idx == sample_limit:
            break

        # Create an input batch
        batch = list()
        for k in range(batch_size):
            if idx + k < len(rgb_paths):
                rgb = cv2.imread(rgb_paths[idx + k])
                depth = cv2.imread(depth_paths[idx + k])
                if img_input_shape:
                    hw_shape = img_input_shape[1:3]  # Get the HW from the input shape
                    wh_shape = hw_shape[::-1]  # Convert HW representation to WH for resizing
                    rgb = cv2.resize(rgb, tuple(wh_shape), interpolation=cv2.INTER_NEAREST)
                    depth = decode_depth(cv2.resize(depth, tuple(wh_shape), interpolation=cv2.INTER_NEAREST))

                rgb = np.asarray(rgb, dtype=np.float32)
                depth = np.asarray(depth, dtype=np.float32)
                batch.append((rgb, depth))
        batches.append(batch)
    return batches


def get_max_classes(x: torch.Tensor) -> torch.Tensor:
    """
    Extracts the most likely class after segmentation
    @param x: a tensor of shape [batchsize, num_classes, height, width]
    @return: a tensor of shape [batchsize, width, height]. Each pixel has the index of the class
    """
    assert x.ndim == 4, f'the ndim should be 4, got shape {x.shape}'
    # get the most likely class
    max_classes = x.max(dim=1).values
    # reshape from [batchsize, height, width] to [batchsize, width, height]
    return torch.permute(max_classes, (0, 2, 1))


if __name__ == '__main__':
    rgbdfolder_2_batches(r'C:\data\UnitreeA1\legrobo_data_1\datastream_1\samples\0', 1, None, 10)
