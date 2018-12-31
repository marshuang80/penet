import torch.nn as nn

from models.layers.unet.unet_conv_block import UNetConvBlock


class UNetEncoder(nn.Module):
    """Encoder (down-sampling layer) for UNet"""
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.pool = nn.Conv2d(in_channels, in_channels, 2, stride=2)
        self.conv = UNetConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)

        return x
