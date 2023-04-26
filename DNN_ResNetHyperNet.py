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
    def __init__(self, emb_dim: int = 64, device: str ='cuda', train: bool = True):
        super(DNN_ResNetHyperNet, self).__init__()

        self.dnn_name = 'DNN_ResNetHyperNet'

        self.emb_dim = emb_dim
        self.device = torch.device(device)

        self.model = Resnet18(num_classes=10).to(self.device)
        model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        model_params = sum(np.prod(p.size()) for p in model_params)
        print(model_params)

        self.HyperNet = HyperNetwork(z_dim=emb_dim).to(self.device)
        model_params = filter(lambda p: p.requires_grad, self.HyperNet.parameters())
        model_params = sum(np.prod(p.size()) for p in model_params)
        print(model_params)

        self.embeddings_sizes = [[4, 4], [4, 4], [4, 4], [4, 4], [8, 4], [8, 8], [8, 8], [8, 8],
                        [16, 8], [16, 16], [16, 16], [16, 16], [32, 16], [32, 32], [32, 32], [32, 32]]
        self.embeddings_list = nn.ModuleList().to(device)
        for i in range(len(self.embeddings_sizes)):
            self.embeddings_list.append(Embedding(z_num=self.embeddings_sizes[i], z_dim=self.emb_dim).to(self.device))
        model_params = filter(lambda p: p.requires_grad, self.embeddings_list.parameters())
        model_params = sum(np.prod(p.size()) for p in model_params)
        print(model_params)

        self.weights = list()
        for idx_emb, emb in enumerate(self.embeddings_list):
            self.weights.append(self.embeddings_list[idx_emb](self.HyperNet).to(self.device))

        self.criterion = nn.CrossEntropyLoss()

        self.params = [{'params': self.HyperNet.parameters()},
                       {'params': self.model.parameters()},
                       {'params': self.embeddings_list.parameters()}]

        self.optimizer = torch.optim.SGD(params=self.params, lr=0.003, momentum=0.9)
        self.lr_scheduler = PolyLR(optimizer=self.optimizer, power=0.9, total_iters=10000)

        self.batch_size = 1
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

        for batch_idx, batch_data in enumerate(self.trainset_dataloader):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            output = self.model(inputs, self.weights)
            loss = self.criterion(output, labels)

            print('Train loss {}/{}: {}'. format(1 + batch_idx, len(self.trainset_dataloader),
                                                 loss.detach().cpu().numpy()))

            # backprop
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            train_loss += loss

        # validation
        print('Validation loop ...')

        self.HyperNet.eval()
        self.embeddings_list.eval()
        self.model.eval()

        for batch_idx, batch_data in enumerate(self.valset_dataloader):
            inputs = batch_data[0].to(self.device)
            labels = batch_data[1].to(self.device)

            output = self.model(inputs, self.weights)
            loss = self.criterion(output, labels)
            val_loss += loss

        if epoch % 50 == 0:
            self.save_model(epoch=epoch, loss=val_loss)

        print("Finished epoch ", epoch)
        print("Training loss: ", train_loss)
        print("Validation loss: ", val_loss)



def main():
    transform = transforms.Compose([
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2023, 0.1994, 0.2010],
                )])

    train_dataset = CIFAR10(train=True, download=True, root='./data', transform=transform)
    test_dataset = CIFAR10(train=False, download=True, root='./data', transform=transform)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dnn = DNN_ResNetHyperNet(device=device)

    model_params = filter(lambda p: p.requires_grad, dnn.parameters())
    model_params = sum(np.prod(p.size()) for p in model_params)
    print(model_params)

    dnn.load_dataset(train_dataset, train_dataset, test_dataset)
    dnn.train()


if __name__ == '__main__':
    main()
