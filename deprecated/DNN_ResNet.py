import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from torchvision.datasets import CIFAR10
from torchvision import transforms


class DNN_ResNetHyperNet(nn.Module):
    def __init__(self, num_classes: int, device: str ='cuda'):
        super(DNN_ResNetHyperNet, self).__init__()

        self.dnn_name = 'DNN_ResNetHyperNet'
        self.device = torch.device(device)

        self.model = torchvision.models.resnet18(weights=None)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=num_classes, bias=True)
        self.model = self.model.to(self.device)
        print(self.model)

        self.criterion = nn.CrossEntropyLoss()

        self.params = [{'params': self.model.parameters()}]

        # self.optimizer = torch.optim.SGD(params=self.params, lr=0.003, momentum=0.9)
        self.optimizer = torch.optim.Adam(params=self.params)
        # self.lr_scheduler = PolyLR(optimizer=self.optimizer, power=0.9)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                       T_max=128, eta_min=1e-4)

        self.batch_size = 64
        self.num_workers = 0
        self.shuffle = True
        self.epochs = 1000

        self.trainset_dataloader = None
        self.valset_dataloader = None
        self.evalset_dataloader = None

    def load_dataset(self, train_dataset, val_dataset, eval_dataset):
        self.trainset_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=self.shuffle,
                                                         batch_size=self.batch_size, num_workers=self.num_workers)
        self.valset_dataloader = torch.utils.data.DataLoader(val_dataset, shuffle=self.shuffle,
                                                       batch_size=self.batch_size, num_workers=self.num_workers)
        self.evalset_dataloader = torch.utils.data.DataLoader(eval_dataset, shuffle=self.shuffle,
                                                        batch_size=self.batch_size, num_workers=self.num_workers)

    def save_model(self, epoch, loss):
        path_to_save = './ckpts/{}-epoch-{}-loss-{}.ckpt'.format(self.dnn_name, epoch, loss)

        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'loss': loss,
            'model_dict': self.model.state_dict(),
        }, path_to_save)

    def load_model(self):
        pass

    def train(self):
        # training
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)

    def train_one_epoch(self, epoch):
        print('')
        print('Training epoch: {}'.format(epoch))

        train_loss = 0
        val_loss = 0

        # training
        self.model.train()

        start_time = time.time()

        for batch_idx, batch_data in enumerate(self.trainset_dataloader):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            output = self.model(inputs)
            loss = self.criterion(output, labels)

            print('Train loss {}/{}: {}'. format(1 + batch_idx, len(self.trainset_dataloader),
                                                 loss.detach().cpu().numpy()))

            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

            gc.collect()

            train_loss += loss.detach().cpu().numpy()

        end_time = time.time()

        # validation
        print('Validation loop ...')

        self.model.eval()

        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valset_dataloader):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                output = self.model(inputs)
                loss = self.criterion(output, labels)
                val_loss += loss.detach().cpu().numpy()

                total += labels.shape[0]
                x = (labels == torch.argmax(output, dim=1)).detach().cpu().numpy()
                correct += np.count_nonzero(x)

                output = None
                gc.collect()

            if epoch % 50 == 0:
                self.save_model(epoch=epoch, loss=val_loss)

            print('correct: ', correct)
            print('total:', total)

            accuracy = 100. * correct / total

            print("Finished epoch ", epoch)
            print("Training took: {}".format(end_time -start_time))
            print("Training loss: ", train_loss)
            print("Validation loss: ", val_loss, "Validation accuracy:", "{}%".format(accuracy))



def main():
    torch.autograd.set_detect_anomaly(True)

    transform = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                )])


    # train_dataset = torch.utils.data.Subset(
    #     CIFAR100(train=True, download=True, root='./data', transform=transform),
    #     torch.arange(0, 1000))
    # test_dataset = torch.utils.data.Subset(
    #     CIFAR100(train=False, download=True, root='./data', transform=transform),
    #     torch.arange(0, 100))

    train_dataset_original = CIFAR10(train=True, download=True, root='./data', transform=transform)
    test_dataset_original = CIFAR10(train=False, download=True, root='./data', transform=transform)

    nc = len(train_dataset_original.classes)

    import random
    train_samples = torch.tensor(random.sample(range(len(train_dataset_original)),
                                               int(len(train_dataset_original) * 0.7)))
    test_samples = torch.tensor(random.sample(range(len(test_dataset_original)),
                                              int(len(test_dataset_original) * 0.2)))

    train_dataset = torch.utils.data.Subset(train_dataset_original, train_samples)
    val_dataset = torch.utils.data.Subset(train_dataset_original, test_samples)
    test_dataset = torch.utils.data.Subset(test_dataset_original, test_samples)

    device = 'cuda'
    dnn = DNN_ResNetHyperNet(device=device, num_classes=nc)

    model_params = filter(lambda p: p.requires_grad, dnn.parameters())
    model_params = sum(np.prod(p.size()) for p in model_params)
    print(model_params)

    dnn.load_dataset(train_dataset, val_dataset, test_dataset)
    dnn.train()


if __name__ == '__main__':
    main()
