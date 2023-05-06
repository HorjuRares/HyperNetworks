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
 * Dataset_SegmentationRGBD.py
 *
 *  Created on: 21.11.2022
 *      Author: Sorin Grigorescu
"""
import os

import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset

# from RovisToolkit.image_display_utils import *
from RovisToolkit.image_display_utils import decode_depth, transform_imagesbatch, imagebatch_2_tensor, \
    transform_tensorbatch, decode_semseg, drawSemanticSegmentation


class Dataset_SegmentationRGBD(Dataset):
    def __init__(self, rovis_databases, device='cuda', transforms=list(), target_transform=None, width=-1, height=-1):
        self.dataset = rovis_databases
        self.transforms = transforms
        self.target_transform = target_transform
        self.device = device
        self.width = width
        self.height = height

        self.rgb_samples_files = list()
        self.depth_samples_files = list()
        self.labels_files = list()

        for dataset in self.dataset:
            for keys_s, keys_l in zip(dataset['keys_samples'], dataset['keys_labels']):
                for i in range(min(len(keys_s), len(keys_l))):
                    s = keys_s[i]
                    l = keys_l[i]
                    desc_s = os.path.join(dataset['path'], "datastream_{}".format(s), "data_descriptor.csv")
                    desc_l = os.path.join(dataset['path'], "datastream_{}".format(l), "data_descriptor.csv")
                    with open(desc_s, "r") as f_s:
                        with open(desc_l, "r") as f_l:
                            lines_samples = f_s.readlines()
                            lines_labels = f_l.readlines()
                    for idx in range(1, len(lines_samples)):
                        try:
                            descriptor_row = lines_samples[idx].split(",")
                            rgb_img_relative_path = descriptor_row[3].strip()
                            depth_img_relative_path = descriptor_row[4].strip()
                        except IndexError:
                            continue
                        rgb_img_path = os.path.join(dataset['path'], "datastream_{}".format(s), rgb_img_relative_path)
                        if not os.path.exists(rgb_img_path) or os.path.isdir(rgb_img_path):
                            continue

                        depth_img_path = os.path.join(dataset['path'], "datastream_{}".format(s), depth_img_relative_path)
                        # if not os.path.exists(depth_img_path) or os.path.isdir(depth_img_path):
                        #     continue

                        try:
                            label_relative_path = lines_labels[idx].split(",")[3]
                        except IndexError:
                            continue
                        label_img_path = os.path.join(dataset['path'], "datastream_{}".format(l), label_relative_path)
                        if not os.path.exists(label_img_path) or os.path.isdir(label_img_path):
                            continue
                        self.rgb_samples_files.append(rgb_img_path)
                        self.depth_samples_files.append(depth_img_path)
                        self.labels_files.append(label_img_path)

    def __getitem__(self, item):
        rgb_img = cv2.imread(self.rgb_samples_files[item])

        rgb_img_copy = rgb_img.copy()

        depth_img_orig = cv2.imread(self.depth_samples_files[item])
        depth_img = decode_depth(depth_img_orig, 16777.216)

        semantic = cv2.imread(self.labels_files[item])
        semantic = semantic[:, :, 0]  # Semantic segmentation images need to be 1-channeled

        if self.width != -1 and self.height != -1:
            rgb_img = np.resize(rgb_img, (self.width, self.height, 3))  # .resize(rgb_img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            depth_img = cv2.resize(depth_img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
            semantic = cv2.resize(semantic, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

        if rgb_img.shape != (320, 320, 3):
            print(rgb_img.shape)
            print('AAAAA')

        rgb_images = [rgb_img]
        rgb_images = transform_imagesbatch(rgb_images, self.transforms)
        rgb_img_tensor = imagebatch_2_tensor(rgb_images, self.device)
        rgb_img_tensor = transform_tensorbatch(rgb_img_tensor, self.transforms)
        # rgb_to_save = rgb_img_tensor[0]

        return {"rgb": rgb_img_tensor[0],
                "depth": depth_img,
                "semantic": semantic}

    def __len__(self):
        return len(self.labels_files)


if __name__ == "__main__":
    pass

    # dataset = Dataset_SegmentationRGBD(rovis_databases=dnn.Datasets_Train,
    #                                    width=dnn.hyperparams['input_shape'][0][2],
    #                                    height=dnn.hyperparams['input_shape'][0][1],
    #                                    transforms=dnn.hyperparams['input_data_transforms'],
    #                                    device=dnn.hyperparams['device'])
    #
    # print('Dataset train has {} sets image-semseg.'.format(len(dataset)))
    #
    # dataloader = DataLoader(dataset,
    #                         batch_size=dnn.hyperparams['batch_size'],
    #                         shuffle=dnn.hyperparams['shuffle'],
    #                         num_workers=dnn.hyperparams['num_workers'])
    #
    # max_imgs_to_disp = 5
    # num_imgs_to_disp = min(max_imgs_to_disp, dnn.hyperparams['batch_size'])
    #
    # for d in dataloader:
    #     imgs_disp = list()
    #     to_be_shown = min(num_imgs_to_disp, d['rgb'].shape[0])
    #     for idx in range(to_be_shown):
    #         rgb_orig = d["rgb"].cpu().detach().numpy()[idx]
    #         depth_orig = d["depth"].cpu().detach().numpy()[idx]
    #         semseg = d["semantic"].cpu().detach().numpy()[idx]
    #         rgb_orig = np.asarray(rgb_orig, dtype=np.float32)
    #         rgb_orig /= 255.0
    #
    #         semseg_out = decode_semseg(semseg, dnn.RGBD_SemSeg_hyp['colormap'])
    #         final = drawSemanticSegmentation(rgb_orig, semseg_out)
    #         final *= 255
    #         final = final.astype(np.uint8)
    #
    #         img_depth = depth_orig.copy() / depth_orig.max()
    #         img_depth = cv2.applyColorMap(np.uint8(img_depth * 255), cv2.COLORMAP_JET)
    #
    #         imgs_disp.append(cv2.vconcat([final, img_depth]))
    #
    #     final_image = cv2.hconcat(imgs_disp)
    #     cv2.imshow("Dataset_Segmentation2D", final_image)
    #     cv2.waitKey(0)
