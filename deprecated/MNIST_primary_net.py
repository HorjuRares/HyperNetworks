import torch
import torch.nn as nn
import torch.nn.functional as F

from hypernetwork_modules import HyperNetwork, Embedding
from utils.ResNetFunctional import ResNetBlock


class MNIST_PRIMARY_NET(nn.Module):
    def __init__(self, num_classes : int = 10):
        super(MNIST_PRIMARY_NET, self).__init__()
        
        self.num_classes = num_classes

        self.hyp_net = HyperNetwork()
        self.embeddings_list = nn.ModuleList()
        self.embeddings_size = [(4, 2), (4, 4), (4, 4), (4, 4)]
        for size_idx, _ in enumerate(self.embeddings_size):
            self.embeddings_list.append(Embedding(z_num=self.embeddings_size[size_idx], z_dim=64))

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.convnet = ResNetBlock(in_channels=32, out_channels=64, downsample=True, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        weights = []
        for emb_idx, _ in enumerate(self.embeddings_list):
            weights.append(self.embeddings_list[emb_idx](self.hyp_net))
        x = self.convnet(x, weights)
        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)

        return x


def main():
    x = torch.rand((1, 3, 28, 28))

    net = MNIST_PRIMARY_NET()

    print(net(x).shape)


if __name__ == '__main__':
    main()