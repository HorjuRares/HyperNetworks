import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import PolynomialLR

from utils.Dataset_SegmentationRGBD import Dataset_SegmentationRGBD
from utils.ModelV2_RGBDNet_Hypernet import Model_RGBDNet_Hypernet
from utils.performance_metrics import *
from RovisToolkit.image_utils import decode_semseg
from RovisToolkit.object_classes import ObjectClasses


NUM_CLASSES = 3


database_train = [{'path': r'C:/dev/Kinect_converted', 'keys_samples': [(1, )], 'keys_labels': [(2, )]}]
database_test = [{'path': r'C:/dev/Kinect_converted', 'keys_samples': [(1, )], 'keys_labels': [(2, )]}]

train_dataset = Dataset_SegmentationRGBD(rovis_databases=database_train,
                                         width=320, height=320)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=0)

test_dataset = Dataset_SegmentationRGBD(rovis_databases=database_test,
                                        width=320, height=320)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=16, num_workers=0)

net = Model_RGBDNet_Hypernet(num_classes=NUM_CLASSES).to('cuda')
loss_fn = torch.nn.NLLLoss(reduction='mean').to('cuda')

epochs = 1000
lr = 0.003

optimizer = optim.Adam(params=net.parameters(), lr=lr, weight_decay=0)
lr_scheduler = PolynomialLR(optimizer=optimizer, total_iters=20000, power=0.9)

colormap = ObjectClasses(r'C:/dev/Kinect_converted/datastream_2/object_classes.conf').colormap()

for epoch in range(epochs):
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

    # after
    # after = list(p.data for p in net.Hypernet.parameters())
    #
    # for idx_p, _ in enumerate(after):
    #     print(torch.equal(after[idx_p], before[idx_p]))

    print('Training loss: {}'.format(training_loss))

    # validation
    print('Validation loop...')

    net.eval()
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(test_dataloader):
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
                # img_rgb_orig = cv2.resize(torch.transpose(batch_data['rgb'][i], (2, 0, 1)).detach().cpu().numpy(),
                #                           (outputs.shape[-2], outputs.shape[-1]))

                semseg_display = decode_semseg(semseg_output, colormap)
                # img_display = cv2.addWeighted(img_rgb_orig.astype(np.float32), 0.6,
                #                               semseg_display.astype(np.float32), 0.5, 0.0)

                cv2.imshow('Validation', semseg_display)
                cv2.waitKey(10)

        print('Finished validation')
        print('Validation loss:', validation_loss)
        print('Validation global:', validation_global / len(test_dataloader))
        print('Validation mean:', validation_mean / len(test_dataloader))
        print('Validation IoU:', validation_IoU / len(test_dataloader))
