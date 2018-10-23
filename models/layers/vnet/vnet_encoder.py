import torch.nn as nn

from .vnet_conv_block import VNetConvBlock
from .vnet_down_sampler import VNetDownSampler


class VNetEncoder(nn.Module):
    """Encoder (down-sampling layer) for VNet"""
    def __init__(self, in_channels, out_channels, num_layers, dropout_prob):
        super(VNetEncoder, self).__init__()
        self.down = VNetDownSampler(in_channels, out_channels, dropout_prob)
        self.conv = VNetConvBlock(num_layers, out_channels)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)  # Note: Skip connection added inside the conv

        return x
