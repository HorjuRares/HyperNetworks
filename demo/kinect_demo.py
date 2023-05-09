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

    checkpoint = torch.load(r'../ckpts/RGBD_Net_weights_epoch_200.pth')
    colormap = ObjectClasses(r'C:/Databases/Kinect_converted/datastream_2/object_classes.conf').colormap()

    dnn = Model_RGBDNet_Hypernet(num_classes=3).to(device_torch)
    before = list(torch.clone(p.data) for p in dnn.depth_embeddings_list.parameters())
    dnn.load_state_dict(checkpoint['model_state_dict'])
    after = list(p.data for p in dnn.depth_embeddings_list.parameters())
    for idx_p, _ in enumerate(after):
        print(torch.equal(after[idx_p], before[idx_p]))

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
            dmap = np.frombuffer(depth_stream.read_frame().get_buffer_as_uint16(),
                                 dtype=np.uint16).reshape(1, 1, 480, 640)
            rgb = np.frombuffer(color_stream.read_frame().get_buffer_as_uint8(),
                                dtype=np.uint8).reshape(1, 480, 640, 3)
            rgb = rgb / np.max(rgb)

            # inference part goes right here
            rgb_tensor = torch.from_numpy(rgb).to(device=device_torch, dtype=torch.float32)
            rgb_tensor = torch.permute(rgb_tensor, dims=(0, 3, 1, 2))
            depth_tensor = torch.from_numpy(dmap / 1000.).to(device=device_torch, dtype=torch.float32)

            input = torch.cat((depth_tensor, rgb_tensor), dim=1)

            start_inference_time = time.time()
            output = dnn(input)
            print('inference took:', time.time() - start_inference_time)

            semseg_out = cv2.resize(decode_semseg(output[0].detach().cpu().numpy(), colormap),
                                    tuple([640, 480]),
                                    interpolation=cv2.INTER_NEAREST)

            cv2.imshow('rgb', rgb[0])
            cv2.imshow('semseg', semseg_out)
            cv2.waitKey(1)

    depth_stream.close()
    color_stream.close()

    openni2.unload()
