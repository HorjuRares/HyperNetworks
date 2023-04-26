import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torchvision.models.resnet import resnet18


class HyperNetwork(nn.Module):
    def __init__(self, f_size = 3, z_dim = 64, out_size=16, in_size=16):
        super(HyperNetwork, self).__init__()
        self.z_dim = z_dim
        self.f_size = f_size
        self.out_size = out_size
        self.in_size = in_size

        # K of size NinFsize x NoutFsize
        # the second layer concatenates the anterior generated weights into K
        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size * self.f_size * self.f_size)), 2))  # wout
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size * self.f_size * self.f_size)), 2))  # bout

        # the first layer generates Nin tensors
        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size * self.z_dim)), 2))  # wi
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size * self.z_dim)), 2))  # bi

    def forward(self, z):
        # where z is zj (the layer embedding)
        h_in = torch.matmul(z, self.w2) + self.b2
        h_in = h_in.view(self.in_size, self.z_dim)

        h_final = torch.matmul(h_in, self.w1) + self.b1
        kernel = h_final.view(self.out_size, self.in_size, self.f_size, self.f_size)

        return kernel


class Embedding(nn.Module):
    def __init__(self, z_num, z_dim):
        super(Embedding, self).__init__()

        self.z_list = nn.ParameterList()
        self.z_num = z_num  # number of embeddings to be concatenated
        # in the paper they consider every filter size to be a multiple of 16, so most probably this is
        # going to be useless
        self.z_dim = z_dim  # embedding dimension

        h, k = self.z_num

        for i in range(h):
            for j in range(k):
                self.z_list.append(Parameter(torch.fmod(torch.randn(self.z_dim), 2)))

    def forward(self, hyper_net):
        ww = []
        h, k = self.z_num

        # compose the resulting Kernel which has the size a multiple of 16
        for i in range(h):
            w = []
            for j in range(k):
                w.append(hyper_net(self.z_list[i*k + j]))
            ww.append(torch.cat(w, dim=1))
        return torch.cat(ww, dim=0)


def main():
    conv_layer = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=2, padding=0, dilation=1, bias=False)
    layer_params = filter(lambda p: p.requires_grad, conv_layer.parameters())
    layer_params = sum(np.prod(p.size()) for p in layer_params)

    print('layer:', layer_params)

    filter_size = conv_layer.kernel_size[0]
    in_ch = conv_layer.in_channels
    out_ch = conv_layer.out_channels

    hyper_net = HyperNetwork(f_size=3, z_dim=64, out_size=16, in_size=16)
    hyp_params = filter(lambda p: p.requires_grad, hyper_net.parameters())
    hyp_params = sum(np.prod(p.size()) for p in hyp_params)
    print('hyp:', hyp_params)

    emb = Embedding(z_num=(1, 1), z_dim=64)
    emb_params = filter(lambda p: p.requires_grad, emb.parameters())
    emb_params = sum(np.prod(p.size()) for p in emb_params)

    print('emb:', emb_params)

    # weights = emb(hyper_net)
    # print(weights.size())
    # conv_layer.weight.data = weights

    model = resnet18(weights=None)
    model_params = filter(lambda p: p.requires_grad, model.parameters())
    model_params = sum(np.prod(p.size()) for p in model_params)

    print(model_params)

    layers = list(model.children())
    for layer in layers:
        print(layer)

    # number_of_conv_params = 0
    # for name, param in model.named_parameters():
    #     if param.requires_grad and 'conv' in name:
    #         print(name, param.shape)
    #         number_of_conv_params += param.shape[0] * param.shape[1] * param.shape[2] * param.shape[3]
    #
    # base_params = model_params - number_of_conv_params
    #
    # embeddings_sizes = [[4, 4], [4, 4], [4, 4], [4, 4], [4, 8], [8, 8], [8, 8], [8, 8],
    #                     [8, 16], [16, 16], [16, 16], [16, 16], [16, 32], [32, 32], [32, 32], [32, 32]]
    # embeddings_list = nn.ModuleList()
    # for i in range(len(embeddings_sizes)):
    #     embeddings_list.append(Embedding(z_num=embeddings_sizes[i], z_dim=64))
    #     emb_params = filter(lambda p: p.requires_grad, embeddings_list[i].parameters())
    #     emb_params = sum(np.prod(p.size()) for p in emb_params)
    #     print(emb_params)
    #     hyp_params += emb_params
    #
    # hyp_params += base_params

    return 0


if __name__ == '__main__':
    main()




