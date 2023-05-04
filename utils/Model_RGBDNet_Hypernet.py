import torch
import torch.nn as nn
import numpy as np

from utils.hypernetwork_modules import HyperNetwork, Embedding
from utils.ResNetFunctional import ResnetForHypernet
from utils.decoding_heads import Model_Decoder_Head


class Model_RGBDNet_Hypernet(nn.Module):
    def __init__(self, num_classes: int = 5):
        super(Model_RGBDNet_Hypernet, self).__init__()

        self.Hypernet = HyperNetwork()

        model_params = filter(lambda p: p.requires_grad, self.Hypernet.parameters())
        model_params = sum(np.prod(p.size()) for p in model_params)
        print('Hypernet params', model_params)

        self.embeddings_sizes = [[4, 4], [4, 4], [4, 4], [4, 4], [8, 4], [8, 8], [8, 8], [8, 8],
                                 [16, 8], [16, 16], [16, 16], [16, 16], [32, 16], [32, 32], [32, 32], [32, 32]]
        self.depth_embeddings_list = nn.ModuleList()
        self.rgb_embeddings_list = nn.ModuleList()

        for emb_idx, _ in enumerate(self.embeddings_sizes):
            self.depth_embeddings_list.append(Embedding(z_num=self.embeddings_sizes[emb_idx], z_dim=64))
            self.rgb_embeddings_list.append(Embedding(z_num=self.embeddings_sizes[emb_idx], z_dim=64))

        model_params = filter(lambda p: p.requires_grad, self.depth_embeddings_list.parameters())
        model_params = sum(np.prod(p.size()) for p in model_params)
        print('Depth embeddings params', model_params)

        self.depth_backbone = ResnetForHypernet(in_channels=1, cached_layers=True)
        self.rgb_backbone = ResnetForHypernet(in_channels=3, cached_layers=True)

        model_params = filter(lambda p: p.requires_grad, self.depth_backbone.parameters())
        model_params = sum(np.prod(p.size()) for p in model_params)
        print('Depth backbone params', model_params)

        self.decoding_head = Model_Decoder_Head(num_classes=num_classes, skip_connections=True)

        model_params = filter(lambda p: p.requires_grad, self.decoding_head.parameters())
        model_params = sum(np.prod(p.size()) for p in model_params)
        print('Decoding head params', model_params)

    def forward(self, x: torch.Tensor):
        x_depth = self.depth_backbone(x=x[:, :1, :, :],
                                      embeddings_list=self.depth_embeddings_list,
                                      hypernet=self.Hypernet)
        x_rgb = self.rgb_backbone(x=x[:, 1:, :, :], embeddings_list=self.rgb_embeddings_list, hypernet=self.Hypernet)

        x = dict()
        for keys in zip(x_rgb.keys(), x_depth.keys()):
            assert keys[0] == keys[1]
            x[keys[0]] = torch.concat(tensors=(x_rgb[keys[0]], x_depth[keys[0]]), dim=1)

        x = self.decoding_head(x)

        return x


def main():
    model = Model_RGBDNet_Hypernet().to('cuda')

    model_params = filter(lambda p: p.requires_grad, model.parameters())
    model_params = sum(np.prod(p.size()) for p in model_params)
    print(model_params)

    x = torch.randn(size=(1, 4, 320, 320)).to('cuda')
    y = model(x)
    print(y.shape)


if __name__ == '__main__':
    main()