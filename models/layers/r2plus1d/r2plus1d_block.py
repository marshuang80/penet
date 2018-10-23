import torch.nn as nn
import torch.nn.functional as F
from .conv_2plus1d import Conv2Plus1D


class R2Plus1DBlock(nn.Module):
    """ResNet simple block using (2+1)D convolution.

    Based on the paper:
    "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    by Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri
    (https://arxiv.org/abs/1711.11248).
    """
    def __init__(self, in_channels, out_channels, down_sample=False,
                 only_spatial_down_sample=False, use_spatial_bn=True, use_bias=True,
                 bn_eps=1e-3, bn_momentum=0.1):
        super(R2Plus1DBlock, self).__init__()

        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.use_bias = use_bias

        if down_sample:
            stride = (1, 2, 2) if only_spatial_down_sample else (2, 2, 2)
        else:
            stride = (1, 1, 1)

        # First (2+1)D conv
        conv_layers = [Conv2Plus1D(in_channels, out_channels, kernel_size=(3, 3, 3), stride=stride)]
        if use_spatial_bn:
            conv_layers += [nn.BatchNorm3d(out_channels, eps=bn_eps, momentum=bn_momentum)]
        conv_layers += [nn.ReLU(inplace=True)]

        # Second (2+1)D conv
        conv_layers += [Conv2Plus1D(out_channels, out_channels, kernel_size=(3, 3, 3))]
        if use_spatial_bn:
            conv_layers += [nn.BatchNorm3d(out_channels, eps=bn_eps, momentum=bn_momentum)]

        skip_layers = []
        if down_sample or in_channels != out_channels:
            # Need transformation for skip connection
            skip_layers += [nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=stride, bias=use_bias)]
            if use_spatial_bn:
                skip_layers += [nn.BatchNorm3d(out_channels, eps=bn_eps, momentum=bn_momentum)]

        self.skip_conn = nn.Sequential(*skip_layers)
        self.conv_block = nn.Sequential(*conv_layers)

    def forward(self, x):
        return F.relu(self.skip_conn(x) + self.conv_block(x), inplace=True)
