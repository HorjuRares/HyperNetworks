import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hypernetwork_modules import Embedding, HyperNetwork


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
        self.relu = nn.ReLU(inplace=True)
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


class ResnetHypernet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, type: str = 'resnet18', ):
        super(ResnetHypernet, self).__init__()

        # self.resnet_block_1 = ResNetBlock(in_channels=64, out_channels=64, stride=1, downsample=False)
        # self.resnet_block_2 = ResNetBlock(in_channels=64, out_channels=128, stride=2, downsample=True)
        # self.resnet_block_3 = ResNetBlock(in_channels=128, out_channels=256, stride=2, downsample=True)
        # self.resnet_block_4 = ResNetBlock(in_channels=256, out_channels=512, stride=2, downsample=True)

        if type == 'resnet18':
            self.resnet_channels = [(64, 64), (64, 128), (128, 256), (256, 512)]
            self.resnet_strides = [1, 2, 2, 2]
            self.resnet_downsampling = [False, True, True, True]

            self.embeddings_sizes = [[4, 4], [4, 4], [4, 4], [4, 4], [8, 4], [8, 8], [8, 8], [8, 8],
                                     [16, 8], [16, 16], [16, 16], [16, 16], [32, 16], [32, 32], [32, 32], [32, 32]]
        else:
            raise NotImplementedError

        self.hypernet = HyperNetwork()

        self.embeddings_list = nn.ModuleList()
        for emb_idx, _ in enumerate(self.embeddings_sizes):
            self.embeddings_list.append(Embedding(z_num=self.embeddings_sizes[emb_idx], z_dim=64))

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.resnet_blocks = nn.ModuleList()
        for block_idx, _ in enumerate(self.resnet_channels):
            in_channels, out_channels = self.resnet_channels[block_idx]
            self.resnet_blocks.append(ResNetBlock(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  stride=self.resnet_strides[block_idx],
                                                  downsample=self.resnet_downsampling[block_idx]))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        for block_idx, _ in enumerate(self.resnet_blocks):
            weights = []
            for emb_idx in range(block_idx * 4, (block_idx + 1) * 4):
                weights.append(self.embeddings_list[emb_idx](self.hypernet))

            x = self.resnet_blocks[block_idx](x, weights)

        x = self.avgpool(x)
        x = x.view(x.shape[0], x.shape[1])
        x = self.fc(x)

        return x


class ResnetForHypernet(nn.Module):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, type: str = 'resnet18', cached_layers: bool = True):
        super(ResnetForHypernet, self).__init__()

        self.cached_layers = cached_layers

        if type == 'resnet18':
            self.resnet_channels = [(64, 64), (64, 128), (128, 256), (256, 512)]
            self.resnet_strides = [1, 2, 2, 2]
            self.resnet_downsampling = [False, True, True, True]
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.resnet_blocks = nn.ModuleList()
        for block_idx, _ in enumerate(self.resnet_channels):
            in_channels, out_channels = self.resnet_channels[block_idx]
            self.resnet_blocks.append(ResNetBlock(in_channels=in_channels,
                                                  out_channels=out_channels,
                                                  stride=self.resnet_strides[block_idx],
                                                  downsample=self.resnet_downsampling[block_idx]))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x: torch.Tensor, embeddings_list: nn.ModuleList, hypernet: nn.Module):
        if self.cached_layers:
            layer_id = 2
            cached_layers = dict()
            x = self.conv1(x)   # 0
            x = self.bn1(x)     # 1
            x = self.relu(x)    # 2
            cached_layers[layer_id] = x
            x = self.pool(x)

            layer_id +=2
            for block_idx, _ in enumerate(self.resnet_blocks):
                weights = []
                for emb_idx in range(block_idx * 4, (block_idx + 1) * 4):
                    weights.append(embeddings_list[emb_idx](hypernet))

                x = self.resnet_blocks[block_idx](x, weights)
                cached_layers[layer_id] = x
                layer_id += 1

            return cached_layers
        else:
            x = self.conv1(x)  # 0
            x = self.bn1(x)  # 1
            x = self.relu(x)  # 2
            x = self.pool(x)

            for block_idx, _ in enumerate(self.resnet_blocks):
                weights = []
                for emb_idx in range(block_idx * 4, (block_idx + 1) * 4):
                    weights.append(embeddings_list[emb_idx](hypernet))

                x = self.resnet_blocks[block_idx](x, weights)

            x = self.avgpool(x)
            x = x.view(x.shape[0], x.shape[1])
            x = self.fc(x)

            return x


class ResNetWithHyperNet(nn.Module):
    def __init__(self, cached_layers: bool = True, num_classes: int = 10):
        super(ResNetWithHyperNet, self).__init__()
        self.cached_layers = cached_layers

        self.hypernet = HyperNetwork()

        self.embeddings_sizes = [[4, 4], [4, 4], [4, 4], [4, 4], [8, 4], [8, 8], [8, 8], [8, 8],
                                 [16, 8], [16, 16], [16, 16], [16, 16], [32, 16], [32, 32], [32, 32], [32, 32]]
        self.embeddings_list = nn.ModuleList()
        for emb_idx, _ in enumerate(self.embeddings_sizes):
            self.embeddings_list.append(Embedding(z_num=self.embeddings_sizes[emb_idx], z_dim=64))

        self.resnet = ResnetForHypernet(cached_layers=self.cached_layers, num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        return self.resnet.forward(x=x, embeddings_list=self.embeddings_list, hypernet=self.hypernet)

def main():
    model = ResNetWithHyperNet(cached_layers=False)
    x = torch.randn(size=(1, 3, 320, 320))
    x = model(x)

    # print(len(x.keys()))
    print(x)


if __name__ == '__main__':
    main()