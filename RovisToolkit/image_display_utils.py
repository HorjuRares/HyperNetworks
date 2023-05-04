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
 * image_display_utils.py
 *
 *  Created on: 27.08.2021
 *      Author: Sorin Grigorescu
"""

import numpy as np
import cv2
import torch

from RovisToolkit.image_utils import *
from RovisToolkit.data_transformations import *


def drawObjectsOnBatches(input_images_batch, output_targets, object_classes, conf_threshold, input_data_transforms=None):
    assert len(input_images_batch) != 0

    # Apply inverse data transforms to input tensor
    if input_data_transforms is not None:
        input_images_batch = untransform_tensorbatch(input_images_batch, input_data_transforms)

    # Convert input tensor to numpy data
    if isinstance(input_images_batch, torch.Tensor):
        input_images_batch = input_images_batch.cpu().float().numpy()
    if isinstance(output_targets, torch.Tensor):
        output_targets = output_targets.cpu().numpy()
    if isinstance(input_images_batch, list):
        input_images_batch = np.asarray(input_images_batch).astype(np.float)
    if isinstance(output_targets, list):
        output_targets = np.asarray(output_targets)

    # Apply inverse data transforms to input numpy data
    if input_data_transforms is not None:
        input_images_batch = untransform_imagesbatch(input_images_batch, input_data_transforms)

    # Scaling
    w = input_images_batch.shape[2]
    h = input_images_batch.shape[1]
    scale = 1
    if (w < 640 or h < 480) and len(output_targets) > 0:
        scale *= 3

    tl = 3
    tf = max(tl - 1, 1)  # font thickness
    colormap = object_classes.colormap()
    batch_plotted = list()
    for i in range(len(input_images_batch)):
        img = input_images_batch[i]
        img = cv2.resize(img, (w * scale, h * scale))

        if len(output_targets) > 0:
            image_targets = output_targets[output_targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6])
            classes = image_targets[:, 1].astype('int')
            labels = image_targets.shape[1] == 6  # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)
            tracking_ids = -1 * torch.ones(classes.shape)  # image_targets[:, 7]

            for j in range(len(boxes)):
                if tracking_ids[j] >= 0:
                    label = '%s %d' % (object_classes.get_name_by_index(int(classes[j])), tracking_ids[j])
                else:
                    label = '%s' % (object_classes.get_name_by_index(int(classes[j])))
                info_conf = '%.2f' % (conf[j]) if conf is not None else ''

                box = boxes[j]
                color = (float(colormap[classes[j]][0]), float(colormap[classes[j]][1]), float(colormap[classes[j]][2]))
                c1, c2 = (int(box[0] * scale), int(box[1] * scale)), (int(box[2] * scale), int(box[3] * scale))
                cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

                sub_img = np.asarray(img[c1[1]:c2[1], c1[0]:c2[0]]).astype(float)
                colored_rect = np.full(sub_img.shape, color, dtype=float)
                res = cv2.addWeighted(sub_img, 0.8, colored_rect, 0.5, 0.2)
                img[c1[1]:c2[1], c1[0]:c2[0]] = res  # Putting the image back to its position

                cv2.rectangle(img, (c1[0] - tl + 1, c1[1] - tf * 18), (c2[0] + tl - 1, c1[1]), color, thickness=-1, lineType=cv2.LINE_AA)
                cv2.putText(img, label, (c1[0], c1[1] - 2), 1, tl * 0.8, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)
                cv2.putText(img, info_conf, (c1[0], c2[1] - 2), 1, tl * 0.8, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)

            img = cv2.resize(img, (w, h))

        batch_plotted.append(img)
    return np.asarray(batch_plotted)


def drawSemanticSegmentation(img, segmentation):
    """ Draw semantic segmentation image """
    # Transpose channels
    if (img.shape[0] <= 3 or img.shape[1] <= 3) and img.shape[2] > 3:
        img = img.transpose(1, 2, 0)
    if (segmentation.shape[0] <= 3 or segmentation.shape[1] <= 3) and segmentation.shape[2] > 3:
        segmentation = segmentation.transpose(1, 2, 0)
    # segmentation = segmentation * 255

    # img = np.array(img, dtype=np.float32)
    if np.max(img) > 2:
        img = img.copy() / np.max(img)

    if np.max(segmentation) > 2:
        segmentation = segmentation.copy() / np.max(segmentation)

    if img.shape != segmentation.shape:
        img = cv2.resize(img, (segmentation.shape[1], segmentation.shape[0]), interpolation=cv2.INTER_NEAREST)

    return cv2.addWeighted(np.asarray(img).astype(np.float32), 0.9, np.asarray(segmentation).astype(np.float32), 0.4, 0.0) * 255.


def drawSemanticSegmentationOnBatches(input_images_batch, output_tensor):
    input_images_batch = nchw_2_nhwc(input_images_batch)

    batch_viz = list()
    for k in range(len(output_tensor)):
        img = drawSemanticSegmentation(input_images_batch[k], output_tensor[k])
        batch_viz.append(img)

    # Convert list to numpy array
    batch_viz = np.asarray(batch_viz)
    return batch_viz

def plot_colormap(object_classes, colormap):
    final_img = np.ones((1, 400, 3), np.uint8) * 255
    for idx in range(object_classes.num_classes):
        cls_name = object_classes.get_name_by_index(idx)
        cls_color = colormap[idx]
        aux_img = np.ones((35, 400, 3), np.uint8) * 255
        aux_img[:, 0:30] = cls_color
        aux_img = cv2.putText(aux_img, cls_name, (35, 27), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        final_img = cv2.vconcat([final_img, aux_img])
    return final_img

def drawPoints():
    """ Draws 2D points """
    pass


def drawVoxels():
    """ Draws 3D voxels """
    pass
