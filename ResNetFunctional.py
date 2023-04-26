import torch
import torch.nn as nn
import torch.nn.functional as F

from hypernetwork_modules import Embedding, HyperNetwork


class Downsample(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super(Downsample, self).__init__(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(num_features=out_channels)
        )


class ResNetBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2, downsample: bool = False):
        super(ResNetBasicBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.downsample = None
        if downsample:
            self.downsample = Downsample(in_channels=in_channels, out_channels=out_channels)

    def forward(self, x: torch.Tensor, w_conv1: torch.Tensor, w_conv2: torch.Tensor):
        residual = x

        x = F.conv2d(input=x, weight=w_conv1, stride=self.stride, padding=1)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.conv2d(input=x, weight=w_conv2, stride=1, padding=1)
        x = self.bn2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)

        return  x


class ResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, downsample: bool):
        super(ResNetBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.downsample = downsample

        self.blocks = nn.Sequential(
            ResNetBasicBlock(in_channels=self.in_channels, out_channels=self.out_channels,
                             stride=self.stride, downsample=self.downsample),
            ResNetBasicBlock(in_channels=self.out_channels, out_channels=self.out_channels, stride=1)
        )

    def forward(self, x: torch.Tensor, weights: list):
        assert len(weights) == 4

        for block_idx, block in enumerate(self.blocks):
            x = block(x, weights[block_idx], weights[block_idx + 1])

        return x


class Resnet18(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super(Resnet18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.LeakyReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnet_block_1 = ResNetBlock(in_channels=64, out_channels=64, stride=1, downsample=False)
        self.resnet_block_2 = ResNetBlock(in_channels=64, out_channels=128, stride=2, downsample=True)
        self.resnet_block_3 = ResNetBlock(in_channels=128, out_channels=256, stride=2, downsample=True)
        self.resnet_block_4 = ResNetBlock(in_channels=256, out_channels=512, stride=2, downsample=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x: torch.Tensor, weights: list):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.resnet_block_1(x, weights[:4])
        x = self.resnet_block_2(x, weights[4:8])
        x = self.resnet_block_3(x, weights[8:12])
        x = self.resnet_block_4(x, weights[12:])

        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)

        return x


def main():
    embeddings_sizes = [[4, 4], [4, 4], [4, 4], [4, 4], [8, 4], [8, 8], [8, 8], [8, 8],
                        [16, 8], [16, 16], [16, 16], [16, 16], [32, 16], [32, 32], [32, 32], [32, 32]]
    embeddings_list = nn.ModuleList()
    for i in range(len(embeddings_sizes)):
        embeddings_list.append(Embedding(z_num=embeddings_sizes[i], z_dim=64))

    hypernetwork = HyperNetwork()

    weights = []
    for idx_emb, emb in enumerate(embeddings_list):
        weights.append(embeddings_list[idx_emb](hypernetwork))
        print(weights[idx_emb].shape)

    model = Resnet18()
    x = torch.randn(size=(1, 3, 320, 320))
    x = model(x, weights)

    print(x.shape)


if __name__ == '__main__':
    main()