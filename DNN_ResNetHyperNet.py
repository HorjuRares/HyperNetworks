import copy
import gc
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import PolynomialLR as PolyLR
from hypernetwork_modules import Embedding, HyperNetwork
from ResNetFunctional import Resnet18
from torchvision.datasets import CIFAR100, CIFAR10
from torchvision import transforms


class DNN_ResNetHyperNet(nn.Module):
    def __init__(self, num_classes: int, emb_dim: int = 64, device: str ='cuda', train: bool = True):
        super(DNN_ResNetHyperNet, self).__init__()

        self.dnn_name = 'DNN_ResNetHyperNet'

        self.emb_dim = emb_dim
        self.device = torch.device(device)

        self.model = Resnet18(num_classes=num_classes).to(self.device)
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        model_params = sum(np.prod(p.size()) for p in model_params)
        print(model_params)

        self.HyperNet = HyperNetwork(z_dim=emb_dim, in_size=16, out_size=16).to(self.device)
        model_params = filter(lambda p: p.requires_grad, self.HyperNet.parameters())
        model_params = sum(np.prod(p.size()) for p in model_params)
        print(model_params)

        self.embeddings_sizes = [[4, 4], [4, 4], [4, 4], [4, 4], [8, 4], [8, 8], [8, 8], [8, 8],
                        [16, 8], [16, 16], [16, 16], [16, 16], [32, 16], [32, 32], [32, 32], [32, 32]]
        # self.embeddings_sizes = [[2, 2], [2, 2], [2, 2], [2, 2], [4, 2], [4, 4], [4, 4], [4, 4],
        #                 [8, 4], [8, 8], [8, 8], [8, 8], [16, 8], [16, 16], [16, 16], [16, 16]]

        self.embeddings_list = nn.ModuleList().to(device)
        for i in range(len(self.embeddings_sizes)):
            self.embeddings_list.append(Embedding(z_num=self.embeddings_sizes[i], z_dim=self.emb_dim).to(self.device))
        model_params = filter(lambda p: p.requires_grad, self.embeddings_list.parameters())
        model_params = sum(np.prod(p.size()) for p in model_params)
        print(model_params)

        self.weights = list()
        # for idx_emb, emb in enumerate(self.embeddings_list):
        #     self.weights.append(self.embeddings_list[idx_emb](self.HyperNet).to(self.device))

        self.criterion = nn.CrossEntropyLoss()

        # self.optimizer = torch.optim.SGD(params=self.params, lr=0.003, momentum=0.9)
        self.model_optimizer = torch.optim.Adam(params=self.model.parameters())
        self.emb_optimizer = torch.optim.Adam(params=self.embeddings_list.parameters())
        self.hyp_optimizer = torch.optim.Adam(params=self.HyperNet.parameters())

        self.model_lr_scheduler = PolyLR(optimizer=self.model_optimizer, power=0.9, total_iters=100)
        self.emb_lr_scheduler = PolyLR(optimizer=self.emb_optimizer, power=0.9, total_iters=100)
        self.hyp_lr_scheduler = PolyLR(optimizer=self.hyp_optimizer, power=0.9, total_iters=100)
        # self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, eta_min=1e-4, T_max=)

        self.batch_size = 64
        self.num_workers = 0
        self.shuffle = True
        self.epochs = 1001

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
            'model_optimizer_state_dict': self.model_optimizer.state_dict(),
            'hyp_optimizer_state_dict': self.hyp_optimizer.state_dict(),
            'emb_optimizer_state_dict': self.emb_optimizer.state_dict(),
            'model_lr_scheduler_state_dict': self.model_lr_scheduler.state_dict(),
            'hyp_lr_scheduler_state_dict': self.hyp_lr_scheduler.state_dict(),
            'emb_lr_scheduler_state_dict': self.emb_lr_scheduler.state_dict(),
            'loss': loss,
            'hypernetwork_state_dict': self.HyperNet.state_dict(),
            'embeddings_state_dict': self.embeddings_list.state_dict(),
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
        self.HyperNet.train()
        self.embeddings_list.train()
        self.model.train()

        start_time = time.time()

        for batch_idx, batch_data in enumerate(self.trainset_dataloader):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            # before = list(torch.clone(p.data.detach()) for p in self.HyperNet.parameters())
            # emb_before = list(torch.clone(p.data.detach()) for p in self.embeddings_list.parameters())
            # model_before = list(torch.clone(p.data.detach()) for p in self.model.parameters())

            for idx_emb, emb in enumerate(self.embeddings_list):
                self.weights.append(self.embeddings_list[idx_emb](self.HyperNet).to(self.device))

            output = self.model(inputs, self.weights)
            loss = self.criterion(output, labels)

            print('Train loss {}/{}: {}'. format(1 + batch_idx, len(self.trainset_dataloader),
                                                 loss.detach().cpu().numpy()))

            # after = list(p.data for p in self.HyperNet.parameters())
            # emb_after = list(p.data for p in self.embeddings_list.parameters())
            # model_after = list(p.data for p in self.model.parameters())

            # backprop
            self.model_optimizer.zero_grad()
            self.emb_optimizer.zero_grad()
            self.hyp_optimizer.zero_grad()


            loss.backward()

            self.model_optimizer.step()
            self.emb_optimizer.step()
            self.hyp_optimizer.step()

            self.model_lr_scheduler.step()
            self.emb_lr_scheduler.step()
            self.hyp_lr_scheduler.step()

            self.weights.clear()
            gc.collect()

            # for i in range(len(after)):
            #     print(torch.equal(after[i], before[i]))

            # for i in range(len(emb_after)):
            #     print(torch.equal(emb_after[i], emb_before[i]))

            # for i in range(len(model_after)):
            #     print(torch.equal(model_after[i], model_before[i]))

            train_loss += loss.detach().cpu().numpy()

        end_time = time.time()

        # validation
        print('Validation loop ...')

        self.HyperNet.eval()
        self.embeddings_list.eval()
        self.model.eval()

        total = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(self.valset_dataloader):
                inputs = batch_data[0].to(self.device)
                labels = batch_data[1].to(self.device)

                for idx_emb, emb in enumerate(self.embeddings_list):
                    self.weights.append(self.embeddings_list[idx_emb](self.HyperNet).to(self.device))

                output = self.model(inputs, self.weights)
                loss = self.criterion(output, labels)
                val_loss += loss.detach().cpu().numpy()

                total += labels.shape[0]
                x = (labels == torch.argmax(output, dim=1)).detach().cpu().numpy()
                correct += np.count_nonzero(x)

                self.weights.clear()
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
                                               # 6 * 64))
    test_samples = torch.tensor(random.sample(range(len(test_dataset_original)),
                                              int(len(test_dataset_original) * 0.2)))
                                              # 2 * 64))

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
