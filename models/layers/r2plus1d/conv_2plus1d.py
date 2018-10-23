import torch.nn as nn


class Conv2Plus1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(1, 1, 1),
                 use_bias=True, use_group_norm=False, max_channels=None):
        """(2+1)D Convolutional Layer.

        (2+1)D convolution is 3D convolution separated into spatial and temporal components
        (with batch norm and ReLU in between).

        Based on the paper:
        "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
        by Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri
        (https://arxiv.org/abs/1711.11248).
        """
        super(Conv2Plus1D, self).__init__()

        # Choose mid_channels to roughly match number of parameters in an analogous Conv3d layer
        i = 3 * in_channels * out_channels * kernel_size[1] * kernel_size[2]
        i /= in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels
        mid_channels = int(i) if max_channels is None else min(int(i), max_channels)

        # Spatial conv, norm, ReLU, temporal conv
        self.spatial_conv = nn.Conv3d(in_channels, mid_channels, kernel_size=(1, kernel_size[1], kernel_size[2]),
                                      stride=(1, stride[1], stride[2]), padding=(0, padding[1], padding[2]),
                                      bias=use_bias)
        self.norm = nn.GroupNorm(mid_channels // 16, mid_channels) if use_group_norm else nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.temporal_conv = nn.Conv3d(mid_channels, out_channels, kernel_size=(kernel_size[0], 1, 1),
                                       stride=(stride[0], 1, 1), padding=(padding[0], 0, 0), bias=use_bias)

    def forward(self, x):
        x = self.spatial_conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.temporal_conv(x)

        return x
