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

epochs = 1001
lr = 0.003

optimizer = optim.Adam(params=net.parameters(), lr=lr, weight_decay=0)
lr_scheduler = PolynomialLR(optimizer=optimizer, total_iters=20000, power=0.9)

colormap = ObjectClasses(r'C:/dev/Kinect_converted/datastream_2/object_classes.conf').colormap()

# load checkpoint
checkpoint = torch.load(r'ckpts/RGBD_Net_weights_Kinect_epoch_100.pth')
net.load_state_dict(checkpoint['model_state_dict'])
net.create_weights()
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1

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
    best_IoU = -1;

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

        # after
        # after = list(p.data for p in net.Hypernet.parameters())
        #
        # for idx_p, _ in enumerate(after):
        #     print(torch.equal(after[idx_p], before[idx_p]))

        print('Training loss: {}'.format(training_loss))

        # validation
        print('Validation loop...')

        net.eval()
        net.create_weights()

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

                    # img_rgb_orig = cv2.resize(imgs_rgb[i].detach().cpu().numpy(), (outputs.shape[-2], outputs.shape[-1]))

                    semseg_display = decode_semseg(semseg_output, colormap)
                    # img_display = cv2.addWeighted(img_rgb_orig.astype(np.float32), 0.6,
                    #                               semseg_display.astype(np.float32), 0.5, 0.0)

                    cv2.imshow('Validation_depth', semseg_display)
                    cv2.waitKey(1)

            # visualize rgb feature maps
            input_rgb = imgs_rgb[0].unsqueeze(dim=0)
            feature_maps_rgb = net.rgb_backbone(input_rgb,
                                            embeddings_list=net.rgb_embeddings_list,
                                            hypernet=net.Hypernet,
                                            weights=net.rgb_backbone_weights)

            input_depth = imgs_depth[0].unsqueeze(dim=0)
            feature_maps_depth = net.depth_backbone(input_depth,
                                            embeddings_list=net.depth_embeddings_list,
                                            hypernet=net.Hypernet,
                                            weights=net.depth_backbone_weights)
            feature_maps_concat = {}

            processed = []
            for feature_map_key in feature_maps_rgb.keys():
                processed.append(extract_feature_maps(feature_maps_rgb[feature_map_key].detach().cpu()))
            for feature_map_key in feature_maps_depth.keys():
                processed.append(extract_feature_maps(feature_maps_depth[feature_map_key].detach().cpu()))
            for feature_map_key in feature_maps_depth.keys():
                feature_maps_concat[feature_map_key] = torch.concat((feature_maps_rgb[feature_map_key],
                                                                     feature_maps_depth[feature_map_key]), dim=1)
                processed.append(extract_feature_maps(feature_maps_concat[feature_map_key].detach().cpu()))

            fig = plt.figure()
            for i in range(len(processed)):
                a = fig.add_subplot(3, 5, i + 1)
                imgplot = plt.imshow(processed[i].numpy())
                a.axis("off")

            plt.savefig(str(r'feature_maps/feature_maps_epoch_{:05}.jpg'.format(epoch)), bbox_inches='tight')
            plt.close(fig)

            # cv2.imshow('feature_maps', img)
            # cv2.waitKey(10)

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

        if epoch % 2 == 0:
            torch.save(
                {
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': running_loss
                },
            r'ckpts/RGBD_Net_weights_Kinect_epoch_{}.pth'.format(epoch))

    print('best_global', best_global)
    print('best_mean', best_mean)
    print('best_IoU', best_IoU)