import torch.nn as nn
import torch.nn.functional as F

class R2Plus1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down_sample=False,
                 only_spatial_down_sample=False, use_spatial_bn=True, use_bias=True,
                 bn_eps=1e-3, bn_momentum=0.9):
        super(R2Plus1DBlock, self).__init__()

        self.bn_eps = bn_eps
        self.bn_momentum = bn_momentum
        self.use_bias = use_bias

        if down_sample:
            stride = (1, 2, 2) if only_spatial_down_sample else (2, 2, 2)
        else:
            stride = (1, 1, 1)

        conv_layers = []

        # First (2+1)D conv
        self._add_2plus1d_conv(conv_layers, in_channels, out_channels,
                               kernel_size=(3, 3, 3), stride=stride, padding=(1, 1, 1))
        if use_spatial_bn:
            conv_layers += [nn.BatchNorm3d(out_channels, eps=bn_eps, momentum=bn_momentum)]
        conv_layers += [nn.ReLU()]

        # Second (2+1)D conv
        self._add_2plus1d_conv(conv_layers, out_channels, out_channels,
                               kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
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

    def _add_2plus1d_conv(self, layers, in_channels, out_channels, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0)):
        """Add R(2+1)D conv block: Spatial conv, batch norm, ReLU, temporal conv."""

        # Compute M_i (from paper: chosen to roughly match number of parameters in a 3D conv)
        i = 3 * in_channels * out_channels * kernel_size[1] * kernel_size[2]
        i /= in_channels * kernel_size[1] * kernel_size[2] + 3 * out_channels
        mid_channels = int(i)

        # Conv over space
        layers += [nn.Conv3d(in_channels, mid_channels, kernel_size=(1, kernel_size[1], kernel_size[2]),
                             stride=(1, stride[1], stride[2]), padding=(0, padding[1], padding[2]), bias=self.use_bias),
                   nn.BatchNorm3d(mid_channels, eps=self.bn_eps, momentum=self.bn_momentum),
                   nn.ReLU()]

        # Conv over time
        layers += [nn.Conv3d(mid_channels, out_channels, kernel_size=(kernel_size[0], 1, 1),
                             stride=(stride[0], 1, 1), padding=(padding[0], 0, 0), bias=self.use_bias)]

    def forward(self, input_):
        return F.relu(self.skip_conn(input_) + self.conv_block(input_))
