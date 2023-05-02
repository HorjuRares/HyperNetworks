from absl import app
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.common_layers import Conv


class Model_Decoder_Head(nn.Module):
    def __init__(
            self,
            num_classes: int,
            backbone_type: str = "resnet18",
            skip_connections: bool = True,
            **kwargs
    ):
        super(Model_Decoder_Head, self).__init__()

        self.num_classes = num_classes
        self.backbone_type = backbone_type
        self.skip_connections = skip_connections

        # Define the decoder layers
        if self.skip_connections is False:
            self.neck = nn.Sequential(Conv(1024, 768),
                                      nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1),
                                      nn.BatchNorm2d(num_features=512),
                                      nn.LeakyReLU(inplace=True))

            self.decoder_01 = Conv(512, 256)
            self.decoder_02 = Conv(256, 128)
            self.decoder_03 = Conv(128, 64)
            self.decoder_04 = Conv(64, 32)
            self.decoder_05 = Conv(32, 16)

            self.final_conv = nn.Conv2d(16, self.num_classes, kernel_size=1)
        else:
            if self.backbone_type == "resnet18":
                self.skip_connection_01 = nn.Sequential(Conv(1024, 768),
                                                        nn.Conv2d(in_channels=768, out_channels=512, kernel_size=1),
                                                        nn.BatchNorm2d(num_features=512),
                                                        nn.LeakyReLU(inplace=True))
                self.decoder_01 = Conv(512, 256)

                self.skip_connection_02 = nn.Sequential(Conv(512, 384),
                                                        nn.Conv2d(in_channels=384, out_channels=256, kernel_size=1),
                                                        nn.BatchNorm2d(num_features=256),
                                                        nn.LeakyReLU(inplace=True))
                self.decoder_02 = Conv(512, 128)

                self.skip_connection_03 = nn.Sequential(Conv(256, 192),
                                                        nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1),
                                                        nn.BatchNorm2d(num_features=128),
                                                        nn.LeakyReLU(inplace=True))
                self.decoder_03 = Conv(256, 64)

                self.skip_connection_04 = nn.Sequential(Conv(128, 96),
                                                        nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1),
                                                        nn.BatchNorm2d(num_features=64),
                                                        nn.LeakyReLU(inplace=True))
                self.decoder_04 = Conv(128, 64)

                self.skip_connection_05 = nn.Sequential(Conv(128, 96),
                                                        nn.Conv2d(in_channels=96, out_channels=64, kernel_size=1),
                                                        nn.BatchNorm2d(num_features=64),
                                                        nn.LeakyReLU(inplace=True))
                self.decoder_05 = Conv(128, 64)

                self.final_conv = nn.Conv2d(64, self.num_classes, kernel_size=1)
            else:
                raise NotImplementedError

        self.ReLU = nn.LeakyReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        if self.skip_connections is False:
            return self.forward_no_skip(x)
        else:
            if self.backbone_type == "resnet18":
                return self.forward_resnet18(x)
            else:
                raise NotImplementedError

    def forward_no_skip(self, x):
        x = self.neck(x)
        x = self.decoder_01(x)
        x = self.upsample(x)
        x = self.decoder_02(x)
        x = self.upsample(x)
        x = self.decoder_03(x)
        x = self.upsample(x)
        x = self.decoder_04(x)
        x = self.upsample(x)
        x = self.decoder_05(x)
        x = self.upsample(x)
        x = self.final_conv(x)

        # Apply softmax activation function to get class probabilities
        x = F.softmax(x, dim=1)

        return x

    def forward_resnet18(self, x):
        r"""x is a dictionary output of a backbone network, where each item is a cached feature map."""
        x1 = self.skip_connection_05(x[2])
        x2 = self.skip_connection_04(x[4])
        x3 = self.skip_connection_03(x[5])
        x4 = self.skip_connection_02(x[6])
        x5 = self.skip_connection_01(x[7])

        x = self.decoder_01(x5)
        x = self.upsample(x)
        x = torch.cat([x, x4], dim=1)

        x = self.decoder_02(x)
        x = self.upsample(x)
        x = torch.cat([x, x3], dim=1)

        x = self.decoder_03(x)
        x = self.upsample(x)
        x = torch.cat([x, x2], dim=1)

        x = self.decoder_04(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)

        x = self.decoder_05(x)
        x = self.upsample(x)

        x = self.final_conv(x)

        # Apply softmax activation function to get class probabilities
        x = nn.LogSoftmax(dim=1)(x)

        return x


def tu_Model_Decoder_Head(_argv):
    backbone_type = 'resnet18'
    skip_connections = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    input_shape = (1, 512, 10, 10)
    num_classes = 5
    model = Model_Decoder_Head(num_classes=num_classes,
                               backbone_type=backbone_type,
                               skip_connections=skip_connections).to(device=device)

    # Mock a backbone's output
    # The input image for the backbone is (1, 3, 320, 320)
    if skip_connections is False:
        x = torch.rand(input_shape, dtype=torch.float32).to(device=device)
    else:
        if backbone_type == 'resnet18':
            x = {
                2: torch.rand((1, 128, 160, 160), dtype=torch.float32).to(device=device),
                4: torch.rand((1, 128, 80, 80), dtype=torch.float32).to(device=device),
                5: torch.rand((1, 256, 40, 40), dtype=torch.float32).to(device=device),
                6: torch.rand((1, 512, 20, 20), dtype=torch.float32).to(device=device),
                7: torch.rand((1, 1024, 10, 10), dtype=torch.float32).to(device=device)
            }
        else:
            raise NotImplementedError

    y = model(x)

    print("Output tensor shape:", y.shape)


if __name__ == '__main__':
    try:
        app.run(tu_Model_Decoder_Head)
    except SystemExit:
        pass
