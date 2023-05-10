import gc
import time
import glob
import numpy as np
import cv2
import os
import open3d as o3d
import torch

from torch.utils.data import DataLoader
from primesense import openni2
from primesense import _openni2 as c_api

from utils.ModelV2_RGBDNet_Hypernet import Model_RGBDNet_Hypernet
from RovisToolkit.image_display_utils import *
from RovisToolkit.object_classes import ObjectClasses
from utils.Dataset_SegmentationRGBD import Dataset_SegmentationRGBD
from RovisToolkit.image_utils import decode_depth


def test_dataset():
    colormap = ObjectClasses(r'C:/Databases/Kinect_converted/datastream_2/object_classes.conf').colormap()
    database_test = [{'path': r'C:/Databases/Kinect_converted', 'keys_samples': [(1,)], 'keys_labels': [(2,)]}]
    test_dataset = Dataset_SegmentationRGBD(rovis_databases=database_test,
                                            width=320, height=320)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=16, num_workers=0)

    checkpoint = torch.load(r'../ckpts/RGBD_Net_weights_epoch_60.pth')
    dnn = Model_RGBDNet_Hypernet(num_classes=3).to('cuda')
    dnn.eval()
    dnn.load_state_dict(checkpoint['model_state_dict'])
    dnn.create_weights()

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
            imgs_rgb = batch_data['rgb'].to(device='cuda', dtype=torch.float32)
            imgs_depth = torch.unsqueeze(batch_data['depth'].to(device='cuda', dtype=torch.float32), dim=1)
            inputs = torch.cat(tensors=(imgs_depth, imgs_rgb), dim=1)

            outputs = dnn(inputs)

            for i in range(outputs.shape[0]):
                semseg_output = outputs[i].detach().cpu().numpy()
                semseg_display = decode_semseg(semseg_output, colormap)

                rgb = np.swapaxes(imgs_rgb[i].detach().cpu().numpy(), axis1=0, axis2=2)
                cv2.imshow('rgb', rgb)

                cv2.imshow('Validation_depth', semseg_display)
                cv2.waitKey(0)

    cv2.destroyAllWindows()


def test_img_folder():
    rgb_path = r'C:/Databases/Kinect_converted/datastream_1/samples/0/left'
    depth_path = r'C:/Databases/Kinect_converted/datastream_1/samples/0/right'

    colormap = ObjectClasses(r'C:/Databases/Kinect_converted/datastream_2/object_classes.conf').colormap()

    checkpoint = torch.load(r'../ckpts/RGBD_Net_weights_epoch_60.pth')
    dnn = Model_RGBDNet_Hypernet(num_classes=3).to('cuda')
    dnn.eval()
    dnn.load_state_dict(checkpoint['model_state_dict'])
    dnn.create_weights()

    with torch.no_grad():
        for path in os.listdir(rgb_path):
            rgb_img = cv2.imread(os.path.join(rgb_path, path))
            depth_img = cv2.imread(os.path.join(depth_path, path))
            depth_img = decode_depth(depth_img, 16777.216)

            rgb_img = cv2.resize(rgb_img, (320, 320), interpolation=cv2.INTER_NEAREST)
            depth_img = cv2.resize(depth_img, (320, 320), interpolation=cv2.INTER_NEAREST)

            rgb_img = np.expand_dims(rgb_img, axis=0).astype(np.float32)
            depth_img = np.expand_dims(depth_img, axis=0)

            if np.max(rgb_img) > 1:
                rgb_img /= 255.

            rgb_img = np.ascontiguousarray(nhwc_2_nchw(rgb_img))

            imgs_rgb = torch.tensor(rgb_img).to(device='cuda', dtype=torch.float32)
            imgs_depth = torch.unsqueeze(torch.tensor(depth_img), dim=1).to(device='cuda', dtype=torch.float32)
            inputs = torch.cat(tensors=(imgs_depth, imgs_rgb), dim=1)

            outputs = dnn(inputs)

            semseg_output = outputs[0].detach().cpu().numpy()
            semseg_display = decode_semseg(semseg_output, colormap)

            cv2.imshow('Validation_depth', semseg_display)
            cv2.waitKey(0)


if __name__ == '__main__':
    # test_dataset()
    test_img_folder()
