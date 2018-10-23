import torch
import torch.nn as nn

from .vnet_conv_block import VNetConvBlock
from .vnet_up_sampler import VNetUpSampler


class VNetDecoder(nn.Module):
    """Decoder (up-sampling layer) for VNet"""
    def __init__(self, in_channels, out_channels, num_layers, dropout_prob):
        super(VNetDecoder, self).__init__()
        self.drop_1 = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else nn.Sequential()
        self.drop_2 = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else nn.Sequential()

        self.up = VNetUpSampler(in_channels, out_channels // 2)
        self.conv = VNetConvBlock(num_layers, out_channels)

    def forward(self, x, x_skip):
        x = self.drop_1(x)
        x_skip = self.drop_2(x_skip)

        x = self.up(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.conv(x)  # Note: Skip connection added inside the conv

        return x
