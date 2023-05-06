import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hypernetwork_modules import *


class Conv_functional(nn.Module):
    def __init__(self, out_channels, padding):
        super(Conv_functional, self).__init__()

        self.padding = padding

        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor, w_conv: torch.Tensor):
        x = F.conv2d(input=x, weight=w_conv, padding=self.padding)
        x = self.bn(x)
        x = self.prelu(x)

        return x

class RGBD_Decoder_functional(nn.Module):
    def __init__(self, num_classes: int, **kwargs):
        super(RGBD_Decoder_functional, self).__init__()

        self.num_classes = num_classes

        self.skip_connection_01 = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1),
                                                nn.BatchNorm2d(num_features=512),
                                                nn.PReLU())
        self.decoder_01 = Conv_functional(out_channels=256, padding=1)

        self.skip_connection_02 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                                                nn.BatchNorm2d(num_features=256),
                                                nn.PReLU())
        self.decoder_02 = Conv_functional(out_channels=128, padding=1)

        self.skip_connection_03 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                                                nn.BatchNorm2d(num_features=128),
                                                nn.PReLU())
        self.decoder_03 = Conv_functional(out_channels=64, padding=1)

        self.skip_connection_04 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64, kernel_size=1),
                                                nn.BatchNorm2d(num_features=64),
                                                nn.PReLU())
        self.decoder_04 = Conv_functional(out_channels=64, padding=1)

        self.skip_connection_05 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=64,  kernel_size=1),
                                                nn.BatchNorm2d(num_features=64),
                                                nn.PReLU())
        self.decoder_05 = Conv_functional(out_channels=64, padding=1)

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=self.num_classes),
            nn.PReLU())
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x: torch.Tensor, embeddings_list: nn.ModuleList, hypernet: nn.Module):
        x1 = self.skip_connection_05(x[2])
        x2 = self.skip_connection_04(x[4])
        x3 = self.skip_connection_03(x[5])
        x4 = self.skip_connection_02(x[6])
        x5 = self.skip_connection_01(x[7])

        weights = list()
        for emb_idx, _ in enumerate(embeddings_list):
            weights.append(embeddings_list[emb_idx](hypernet))

        x = self.decoder_01(x5, w_conv=weights[0])
        x = self.upsample(x)
        x = torch.cat([x, x4], dim=1)

        x = self.decoder_02(x, w_conv=weights[1])
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)

        x = self.decoder_03(x, w_conv=weights[2])
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)

        x = self.decoder_04(x, w_conv=weights[3])
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)

        x = self.decoder_05(x, w_conv=weights[4])
        x = self.upsample(x)

        x = self.final_conv(x)
        x = nn.LogSoftmax(dim=1)(x)

        return x


if __name__ == '__main__':
    model = RGBD_Decoder_functional(num_classes=5)

    hypernet = HyperNetwork()

    # output, input
    embeddings_sizes = [[16, 32], [8, 32], [4, 16], [4, 8], [4, 8]]
    embeddings_list = nn.ModuleList()
    for emb_idx, _ in enumerate(embeddings_sizes):
        embeddings_list.append(Embedding(z_num=embeddings_sizes[emb_idx], z_dim=64))

    x = {
        2: torch.rand((1, 128, 160, 160), dtype=torch.float32),
        4: torch.rand((1, 128, 80, 80), dtype=torch.float32),
        5: torch.rand((1, 256, 40, 40), dtype=torch.float32),
        6: torch.rand((1, 512, 20, 20), dtype=torch.float32),
        7: torch.rand((1, 1024, 10, 10), dtype=torch.float32)
    }

    print(model(x, embeddings_list, hypernet))
