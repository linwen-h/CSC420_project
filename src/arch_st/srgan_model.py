from layers.linear import *
from layers.conv2d import *
from base_model import StnModel
from utils.dropout_utils import *

import torch.nn.functional as F
import torch.nn as nn
import torch

class StnSRResNet(StnModel):
    def __init__(self, num_classes, num_hyper, h_container, in_channel=3, n_feats=64,
                 kernel_size=3, num_blocks=16, activation=nn.PReLU(), scale=4):
        super(StnSRResNet, self).__init__()
        self.num_hyper = num_hyper
        self.h_container = h_container

        self.conv1 = [
            StnConv2d(in_channel, n_feats, kernel_size, num_hyper,
                      padding=kernel_size//2, stride=1, bias=True),
            activation
        ]
        self.res_blocks = [[
            StnConv2d(n_feats, n_feats, kernel_size, num_hyper,
                      padding=kernel_size//2, stride=1, bias=True),
            nn.BatchNorm2d(n_feats),
            activation,
            StnConv2d(n_feats, n_feats, kernel_size, num_hyper,
                      padding=kernel_size//2, stride=1, bias=True),
            nn.BatchNorm2d(n_feats)
        ] for _ in range(num_blocks)]

        self.conv2 = [
            StnConv2d(n_feats, n_feats, kernel_size, num_hyper,
                      padding=kernel_size//2, stride=1, bias=True),
            nn.BatchNorm2d(n_feats)
        ]

        if scale == 4:
            self.upsample_blocks = [
                StnConv2d(n_feats, n_feats * scale, kernel_size, num_hyper,
                          padding=kernel_size//2, stride=1, bias=True),
                nn.PixelShuffle(2),
                activation,
                StnConv2d(n_feats, n_feats * scale, kernel_size, num_hyper,
                          padding=kernel_size//2, stride=1, bias=True),
                nn.PixelShuffle(2),
                activation
            ]
        else:
            self.upsample_blocks = [
                StnConv2d(n_feats, n_feats * scale * scale, kernel_size, num_hyper,
                          padding=kernel_size//2, stride=1, bias=True),
                nn.PixelShuffle(2),
                activation
            ]

        self.conv3 = [
            StnConv2d(n_feats, in_channel, kernel_size, num_hyper,
                      padding=kernel_size//2, stride=1, bias=True),
            nn.Tanh()
        ]

        def flatten(t): return [item for sublist in t for item in sublist]

        self.layers = nn.ModuleList([
            *self.conv1, *flatten(self.res_blocks), *self.conv2,
            *self.upsample_blocks, *self.conv3
        ])

    def get_layers(self):
        return self.layers

    def forward(self, x, h_net, h_param):
        # if "dropout0" in self.h_container.h_dict:
        #     x = dropout_2d(x, self.h_container.transform_perturbed_hyper(
        #         h_param, "dropout0"), self.training)

        x = self.conv1[0](x, h_net)
        x = self.conv1[1](x)
        # if "dropout1" in self.h_container.h_dict:
        #     x = dropout_2d(x, self.h_container.transform_perturbed_hyper(
        #         h_param, "dropout1"), self.training)

        _skip_connection = x
        for c1, bn1, act, c2, bn2 in self.res_blocks:
            res = c1(x, h_net)
            res = act(bn1(res))
            res = c2(res, h_net)
            res = bn2(res)
            x = x + res
        x = self.conv2[0](x, h_net)
        x = self.conv2[1](x)
        x = x + _skip_connection

        for layer in self.upsample_blocks:
            if isinstance(layer, StnConv2d):
                x = layer(x, h_net)
            else:
                x = layer(x)

        x = self.conv3[0](x, h_net)
        x = self.conv3[1](x)

        return x



class Discriminator(nn.Module):
    def __init__(self, img_feat=3, n_feats=64, kernel_size=3, act=nn.LeakyReLU(inplace=True), num_of_block=3,
                 patch_size=96):
        super(Discriminator, self).__init__()
        self.act = act
        self.conv01 = conv(in_channel=img_feat, out_channel=n_feats, kernel_size=3, BN=False, act=self.act)
        self.conv02 = conv(in_channel=n_feats, out_channel=n_feats, kernel_size=3, BN=False, act=self.act, stride=2)

        body = [discrim_block(in_feats=n_feats * (2 ** i), out_feats=n_feats * (2 ** (i + 1)), kernel_size=3, act=self.act) for i in range(num_of_block)]
        self.body = nn.Sequential(*body)

        self.linear_size = ((patch_size // (2 ** (num_of_block + 1))) ** 2) * (n_feats * (2 ** num_of_block))

        tail = [nn.Linear(self.linear_size, 1024), self.act, nn.Linear(1024, 1), nn.Sigmoid()]
        self.tail = nn.Sequential(*tail)

    def forward(self, x):
        x = self.conv01(x)
        x = self.conv02(x)
        x = self.body(x)
        x = x.view(-1, self.linear_size)
        x = self.tail(x)
        return x
