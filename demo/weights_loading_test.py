import gc
import time
import glob
import numpy as np
import cv2
import os
import open3d as o3d
import torch
import onnx
import onnxruntime

from torch.utils.data import DataLoader
from primesense import openni2
from primesense import _openni2 as c_api

from utils.ModelV2_RGBDNet_Hypernet import Model_RGBDNet_Hypernet
from RovisToolkit.image_display_utils import *
from RovisToolkit.object_classes import ObjectClasses
from utils.Dataset_SegmentationRGBD import Dataset_SegmentationRGBD
from RovisToolkit.image_utils import decode_depth
from torchvision.models.segmentation import deeplabv3_resnet50
from utils.performance_metrics import *


def test_dataset():
    colormap = ObjectClasses(r'C:/Databases/CarlaDepthTest/datastream_4/object_classes.conf').colormap()
    database_test = [{'path': r'C:/Databases/CarlaDepthTest', 'keys_samples': [(1,)], 'keys_labels': [(4,)]}]
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


def test_onnx_prediction():
    rgb_path = r'C:/Databases/Kinect_converted/datastream_1/samples/0/left'
    depth_path = r'C:/Databases/Kinect_converted/datastream_1/samples/0/right'
    colormap = ObjectClasses(r'C:/Databases/Kinect_converted/datastream_2/object_classes.conf').colormap()

    model = onnx.load(r'DNN_DeepLab_Hypernet_cpu.onnx')

    onnx.checker.check_model(model)
    onnx_inference_session = onnxruntime.InferenceSession(r'DNN_DeepLab_Hypernet_cpu.onnx',
                                                          providers=['CPUExecutionProvider'])
    inputs = onnx_inference_session.get_inputs()

    inference_time = 0

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

        imgs_rgb = torch.tensor(rgb_img).to(dtype=torch.float32)
        imgs_depth = torch.unsqueeze(torch.tensor(depth_img), dim=1).to(dtype=torch.float32)
        input = torch.cat(tensors=(imgs_depth, imgs_rgb), dim=1)

        start_time = time.time()
        outputs = onnx_inference_session.run(None, {inputs[0].name: imgs_rgb.detach().numpy()})
        inference_time += time.time() - start_time

        semseg_display = decode_semseg(outputs[0][0], colormap)

        cv2.imshow('Validation_depth', semseg_display)
        cv2.waitKey(0)

    print('Average inference time:', inference_time / len(os.listdir(rgb_path)))


def test_dataset_deep():
    best_global = -1
    best_mean = -1
    best_IoU = -1

    NUM_CLASSES=3
    colormap = ObjectClasses(r'C:/Databases/CarlaDepthTest/datastream_4/object_classes.conf').colormap()

    database_test = [{'path': r'C:/Databases/CarlaDepthTest', 'keys_samples': [(1,)], 'keys_labels': [(4,)]}]
    test_dataset = Dataset_SegmentationRGBD(rovis_databases=database_test,
                                            width=320, height=320)
    test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=4, num_workers=0)

    dnn = deeplabv3_resnet50(weights=None, num_classes=NUM_CLASSES).to('cpu')
    loss_fn = torch.nn.CrossEntropyLoss(reduction='mean').to('cpu')
    optimizer = torch.optim.Adam(params=dnn.parameters(), lr=0.003, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer=optimizer, total_iters=20000, power=0.9)

    for epoch in range(0, 1001):
        training_loss = 0
        validation_loss = 0

        validation_global = 0
        validation_mean = 0
        validation_IoU = 0

        # training
        print('Training epoch {}'.format(epoch))

        dnn.train()

        for batch_idx, batch_data in enumerate(test_dataloader):
            imgs_rgb = batch_data['rgb'].to(device='cpu', dtype=torch.float32)
            imgs_depth = torch.unsqueeze(batch_data['depth'].to(device='cpu', dtype=torch.float32), dim=1)
            labels = batch_data['semantic'].to('cpu').long()
            inputs = torch.cat(tensors=(imgs_depth, imgs_rgb), dim=1)

            outputs = dnn(imgs_rgb)
            loss = loss_fn(outputs['out'], labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss = loss.detach().cpu().numpy()
            training_loss += running_loss

            print('{}/{}: {}'.format(batch_idx + 1, len(test_dataloader), running_loss))

        print('Training loss: {}'.format(training_loss))

        # validation
        print('Validation loop...')

        dnn.eval()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(test_dataloader):
                imgs_rgb = batch_data['rgb'].to(device='cpu', dtype=torch.float32)
                imgs_depth = torch.unsqueeze(batch_data['depth'].to(device='cpu', dtype=torch.float32), dim=1)
                labels = batch_data['semantic'].to('cpu').long()
                inputs = torch.cat(tensors=(imgs_depth, imgs_rgb), dim=1)

                outputs = dnn(imgs_rgb)
                loss = loss_fn(outputs['out'], labels)

                running_loss = loss.detach().cpu().numpy()
                validation_loss += running_loss

                validation_global += metrics_global_accuracy(ground_truth=torch.unsqueeze(labels, dim=1),
                                                             prediction=outputs['out'])
                validation_mean += metrics_mean_accuracy(ground_truth=torch.unsqueeze(labels, dim=1),
                                                         prediction=outputs['out'], num_classes=NUM_CLASSES)
                validation_IoU += metrics_IoU(ground_truth=torch.unsqueeze(labels, dim=1),
                                              prediction=outputs['out'], num_classes=NUM_CLASSES)

                for i in range(outputs['out'].shape[0]):
                    semseg_output = outputs['out'][i].detach().cpu().numpy()

                    # img_rgb_orig = cv2.resize(imgs_rgb[i].detach().cpu().numpy(), (outputs.shape[-2], outputs.shape[-1]))

                    semseg_display = decode_semseg(semseg_output, colormap)
                    # img_display = cv2.addWeighted(img_rgb_orig.astype(np.float32), 0.6,
                    #                               semseg_display.astype(np.float32), 0.5, 0.0)

                    # cv2.imshow('Validation_rgb', img_rgb_orig)
                    cv2.imshow('Validation_depth', semseg_display)
                    cv2.waitKey(1)

            print('Finished validation')
            print('Validation loss:', validation_loss)
            print('Validation global:', validation_global / len(test_dataloader))
            print('Validation mean:', validation_mean / len(test_dataloader))
            print('Validation IoU:', validation_IoU / len(test_dataloader))

            if validation_global / len(test_dataloader):
                best_global = validation_global / len(test_dataloader)
            if validation_mean / len(test_dataloader):
                best_mean = validation_mean / len(test_dataloader)
            if validation_IoU / len(test_dataloader):
                best_IoU = validation_IoU / len(test_dataloader)

            torch.onnx.export(dnn,
                              imgs_rgb,
                              'DNN_DeepLab_cpu.onnx',
                              opset_version=12,
                              input_names=["input"],
                              output_names=["output"],
                              dynamic_axes={"input": {0: "batch_size"},
                                            "output": {0: "batch_size"}})

    print('best_global', best_global)
    print('best_mean', best_mean)
    print('best_IoU', best_IoU)


if __name__ == '__main__':
    test_dataset()
    # test_img_folder()
    # test_dataset_deep()
    # test_onnx_prediction()