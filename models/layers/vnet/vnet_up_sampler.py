import torch.nn as nn


class VNetUpSampler(nn.Module):
    """Up-sampling 3D transpose conv and group norm."""
    def __init__(self, in_channels, out_channels):
        super(VNetUpSampler, self).__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(out_channels // 16, out_channels)
        self.relu = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)

        return x
