import torch.nn as nn


class VNetDownSampler(nn.Module):
    """Down-sampling 3D conv with group norm."""
    def __init__(self, in_channels, out_channels, dropout_prob):
        super(VNetDownSampler, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = nn.GroupNorm(out_channels // 16, out_channels)
        self.relu = nn.PReLU(out_channels)
        self.drop = nn.Dropout3d(dropout_prob) if dropout_prob > 0 else nn.Sequential()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.drop(x)

        return x
