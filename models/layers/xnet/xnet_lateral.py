import torch.nn as nn


class XNetLateral(nn.Module):
    """Lateral connection layer for XNet."""
    def __init__(self, in_channels, out_channels):
        super(XNetLateral, self).__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(out_channels // 16, out_channels)
        self.relu = nn.LeakyReLU()

    def forward(self, x, x_skip):
        # Reduce number of channels in skip connection
        x_skip = self.conv(x_skip)
        x_skip = self.norm(x_skip)
        x_skip = self.relu(x_skip)

        # Add reduced feature map
        x += x_skip

        return x
