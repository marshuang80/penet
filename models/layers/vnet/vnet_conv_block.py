import torch.nn as nn


class VNetConvBlock(nn.Module):
    """Residual conv block for VNet

    Input channels match output channels. Output is sum of:
      - num_blocks x (Conv3d, GroupNorm, PReLU)
      - skip connection
      """
    def __init__(self, num_layers, num_channels, kernel_size=5, stride=1, padding=2):
        super(VNetConvBlock, self).__init__()

        self.num_channels = num_channels

        conv_layers = []
        for _ in range(num_layers):
            conv_layers += [nn.Conv3d(num_channels, num_channels, kernel_size, stride=stride, padding=padding),
                            nn.GroupNorm(num_channels // 16, num_channels),
                            nn.PReLU(num_channels)]
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        if x.size(1) != self.num_channels:
            # Tile along channel dimension to match (note: `expand` does not actually copy)
            x = x.expand(-1, self.num_channels // x.size(1), -1, -1, -1)

        x = self.conv(x) + x

        return x
