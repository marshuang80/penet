import torch
import torch.nn as nn

from models.layers.unet.unet_conv_block import UNetConvBlock
from .unet_copy_crop import UNetCopyCrop


class UNetDecoder(nn.Module):
    """Decoder (up-sampling layer) for UNet"""
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.copy_crop = UNetCopyCrop()
        self.conv = UNetConvBlock(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, x_skip):
        x = self.up(x)
        x = torch.cat([x_skip, x], dim=1)  # self.copy_crop(x, x_skip)
        x = self.conv(x)

        return x
