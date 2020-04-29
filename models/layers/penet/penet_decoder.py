import torch.nn as nn

from models.layers.penet.penet_lateral import PENetLateral


class PENetDecoder(nn.Module):
    """Decoder (up-sampling layer) for PENet"""
    def __init__(self, skip_channels, in_channels, mid_channels, out_channels, kernel_size=4, stride=2):
        super(PENetDecoder, self).__init__()

        if skip_channels > 0:
            self.lateral = PENetLateral(skip_channels, in_channels)

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu1 = nn.LeakyReLU()

        self.conv2 = nn.ConvTranspose3d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=1)
        self.norm2 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu2 = nn.LeakyReLU()

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm3 = nn.GroupNorm(out_channels // 16, out_channels)
        self.relu3 = nn.LeakyReLU()

    def forward(self, x, x_skip=None):
        if x_skip is not None:
            x = self.lateral(x, x_skip)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x
