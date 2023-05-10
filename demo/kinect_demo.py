import gc
import time

import numpy as np
import cv2
import os
import open3d as o3d
import torch
from primesense import openni2
from primesense import _openni2 as c_api

from utils.ModelV2_RGBDNet_Hypernet import Model_RGBDNet_Hypernet
from RovisToolkit.image_display_utils import *
from RovisToolkit.object_classes import ObjectClasses


if __name__ == '__main__':
    # main()
    device_torch = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device_torch = 'cpu'

    checkpoint = torch.load(r'../ckpts/RGBD_Net_weights_epoch_60.pth')
    colormap = ObjectClasses(r'C:/Databases/Kinect_converted/datastream_2/object_classes.conf').colormap()

    dnn = Model_RGBDNet_Hypernet(num_classes=3).to(device_torch)
    dnn.load_state_dict(checkpoint['model_state_dict'])
    dnn.eval()
    dnn.create_weights()

    openni2.initialize('./Redist')

    device = openni2.Device.open_any()
    depth_stream = device.create_depth_stream()
    color_stream = device.create_color_stream()
    depth_stream.set_video_mode(
        c_api.OniVideoMode(pixelFormat=c_api.OniPixelFormat.ONI_PIXEL_FORMAT_DEPTH_1_MM, resolutionX=640,
                           resolutionY=480, fps=30))
    depth_stream.start()
    color_stream.start()

    with torch.no_grad():
        while True:
            depth_img = np.frombuffer(depth_stream.read_frame().get_buffer_as_uint16(),
                                      dtype=np.uint16).reshape(480, 640) / 1000.
            rgb_img = np.frombuffer(color_stream.read_frame().get_buffer_as_uint8(),
                                    dtype=np.uint8).reshape(480, 640, 3)

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

            start_inference_time = time.time()
            outputs = dnn(inputs)
            print('inference took:', time.time() - start_inference_time)

            semseg_out = cv2.resize(decode_semseg(outputs[0].detach().cpu().numpy(), colormap),
                                    tuple([640, 480]),
                                    interpolation=cv2.INTER_NEAREST)

            cv2.imshow('semseg', semseg_out)
            cv2.waitKey(1)

    depth_stream.close()
    color_stream.close()

    openni2.unload()
