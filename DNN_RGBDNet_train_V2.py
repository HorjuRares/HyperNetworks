import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt
import PIL.Image as Image

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR

from utils.Dataset_SegmentationRGBD import Dataset_SegmentationRGBD
from utils.ModelV2_RGBDNet_Hypernet import Model_RGBDNet_Hypernet
from utils.performance_metrics import *
from RovisToolkit.image_utils import decode_semseg
from RovisToolkit.object_classes import ObjectClasses


NUM_CLASSES = 3

np.random.seed(20000804)


# database_train = [{'path': r'C:/Databases/Kinect_converted', 'keys_samples': [(1, )], 'keys_labels': [(2, )]}]
# database_test = [{'path': r'C:/Databases/Kinect_converted', 'keys_samples': [(1, )], 'keys_labels': [(2, )]}]
database = [{'path': r'C:/Databases/Kinect_converted', 'keys_samples': [(1, )], 'keys_labels': [(2, )]}]
dataset = Dataset_SegmentationRGBD(rovis_databases=database, width=320, height=320)

indices = np.arange(len(dataset))

train_indices = np.random.choice(a=indices, size=int(0.7 * len(dataset)), replace=False)
indices = np.setdiff1d(ar1=indices, ar2=train_indices)
train_dataset = torch.utils.data.Subset(dataset, train_indices)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=0)

val_indices = np.random.choice(a=indices, size=int(0.667 * len(indices)), replace=False)
indices = np.setdiff1d(ar1=indices, ar2=val_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=16, num_workers=0)

eval_dataset = torch.utils.data.Subset(dataset, indices)
eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=16, num_workers=0)

net = Model_RGBDNet_Hypernet(num_classes=NUM_CLASSES).to('cuda')
loss_fn = torch.nn.NLLLoss(reduction='mean').to('cuda')

epochs = 1001
lr = 0.003

optimizer = optim.Adam(params=net.parameters(), lr=lr, weight_decay=0)
lr_scheduler = PolynomialLR(optimizer=optimizer, total_iters=20000, power=0.9)

colormap = ObjectClasses(r'C:/Databases/Kinect_converted/datastream_2/object_classes.conf').colormap()

start_epoch = 0

# load checkpoint

# checkpoint = torch.load(r'ckpts/RGBD_Net_weights_Carla_epoch_72.pth')
# net.load_state_dict(checkpoint['model_state_dict'])
# net.create_weights()
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
# start_epoch = checkpoint['epoch'] + 1

# export the model to onnx

# assert checkpoint is not None or len(checkpoint.keys()) > 0
#
# channels = 4
# in_tensor_width = 320
# in_tensor_height = 320
# x = torch.rand(16, channels, in_tensor_height, in_tensor_width).to('cuda')
#
# torch.onnx.export(net,
#                   x,
#                   'DNN_RGBDNet_Hypernet_cuda.onnx',
#                   opset_version=12,
#                   input_names=["input"],
#                   output_names=["output"],
#                   dynamic_axes={"input": {0: "batch_size"},
#                                 "output": {0: "batch_size"}})


def extract_feature_maps(feature_map: torch.Tensor):
    feature_map = feature_map.squeeze(0)
    gray_scale = torch.sum(feature_map, 0)
    gray_scale = gray_scale / feature_map.shape[0]

    return gray_scale


if __name__ == '__main__':
    best_global = -1
    best_mean = -1
    best_IoU = -1

    for epoch in range(start_epoch, epochs):
        training_loss = 0
        validation_loss = 0

        validation_global = 0
        validation_mean = 0
        validation_IoU = 0

        # training
        print('Training epoch {}'.format(epoch))

        net.train()

        # before
        # before = list(torch.clone(p.data) for p in net.Hypernet.parameters())

        for batch_idx, batch_data in enumerate(train_dataloader):
            imgs_rgb = batch_data['rgb'].to(device='cuda', dtype=torch.float32)
            imgs_depth = torch.unsqueeze(batch_data['depth'].to(device='cuda', dtype=torch.float32), dim=1)
            labels = batch_data['semantic'].to('cuda').long()
            inputs = torch.cat(tensors=(imgs_depth, imgs_rgb), dim=1)

            outputs = net(inputs)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            running_loss = loss.detach().cpu().numpy()
            training_loss += running_loss


            print('{}/{}: {}'.format(batch_idx + 1, len(train_dataloader), running_loss))

        print('Training loss: {}'.format(training_loss))

        # validation
        print('Validation loop...')

        net.eval()
        net.create_weights()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_dataloader):
                imgs_rgb = batch_data['rgb'].to(device='cuda', dtype=torch.float32)
                imgs_depth = torch.unsqueeze(batch_data['depth'].to(device='cuda', dtype=torch.float32), dim=1)
                labels = batch_data['semantic'].to('cuda').long()
                inputs = torch.cat(tensors=(imgs_depth, imgs_rgb), dim=1)

                outputs = net(inputs)
                loss = loss_fn(outputs, labels)

                running_loss = loss.detach().cpu().numpy()
                validation_loss += running_loss

                validation_global += metrics_global_accuracy(ground_truth=torch.unsqueeze(labels, dim=1),
                                                             prediction=outputs)
                validation_mean += metrics_mean_accuracy(ground_truth=torch.unsqueeze(labels, dim=1),
                                                         prediction=outputs, num_classes=NUM_CLASSES)
                validation_IoU += metrics_IoU(ground_truth=torch.unsqueeze(labels, dim=1),
                                              prediction=outputs, num_classes=NUM_CLASSES)

                for i in range(outputs.shape[0]):
                    semseg_output = outputs[i].detach().cpu().numpy()

                    img_rgb_orig = imgs_rgb[i].detach().cpu().numpy()
                    img_rgb_orig = cv2.cvtColor(np.transpose(a=img_rgb_orig, axes=(1, 2, 0)), cv2.COLOR_BGR2RGB)

                    semseg_display = decode_semseg(semseg_output, colormap)

                    img_display = cv2.addWeighted(img_rgb_orig.astype(np.float32), 0.6,
                                                  semseg_display.astype(np.float32), 0.5, 0.0)

                    cv2.imshow('Validation', img_display)
                    cv2.waitKey(1)

            print('Finished validation')
            print('Validation loss:', validation_loss)
            print('Validation global:', validation_global / len(val_dataloader))
            print('Validation mean:', validation_mean / len(val_dataloader))
            print('Validation IoU:', validation_IoU / len(val_dataloader))

            if validation_global / len(val_dataloader):
                best_global = validation_global / len(val_dataloader)
            if validation_mean / len(val_dataloader):
                best_mean = validation_mean / len(val_dataloader)
            if validation_IoU / len(val_dataloader):
                best_IoU = validation_IoU / len(val_dataloader)

        if epoch % 10 == 0:
            torch.save(
                {
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': running_loss
                },
            r'ckpts/RGBD_Net_weights_KinectSCSS_epoch_{}.pth'.format(epoch))

    print('best_global', best_global)
    print('best_mean', best_mean)
    print('best_IoU', best_IoU)