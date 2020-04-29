import torch
import torch.nn as nn


class PENetASPPool(nn.Module):
    """Atrous Spatial Pyramid Pooling layer.

    Based on the paper:
    "Rethinking Atrous Convolution for Semantic Image Segmentation"
    by Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
    (https://arxiv.org/abs/1706.05587).
    """
    def __init__(self, in_channels, out_channels):
        super(PENetASPPool, self).__init__()

        self.mid_channels = out_channels // 4
        self.in_conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2),
                                     nn.GroupNorm(out_channels // 16, out_channels),
                                     nn.LeakyReLU(inplace=True))

        self.conv1 = nn.Conv3d(out_channels, self.mid_channels, kernel_size=1)
        self.conv2 = nn.Conv3d(out_channels, self.mid_channels, kernel_size=3, padding=6, dilation=6)
        self.conv3 = nn.Conv3d(out_channels, self.mid_channels, kernel_size=3, padding=12, dilation=12)
        self.conv4 = nn.Sequential(nn.AdaptiveAvgPool3d(1),
                                   nn.Conv3d(out_channels, self.mid_channels, kernel_size=1))
        self.norm = nn.GroupNorm(out_channels // 16, out_channels)
        self.relu = nn.LeakyReLU(inplace=True)

        self.out_conv = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=1),
                                      nn.GroupNorm(out_channels // 16, out_channels),
                                      nn.LeakyReLU(inplace=True))

    def forward(self, x):
        x = self.in_conv(x)

        # Four parallel paths with different dilation factors
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_3 = self.conv3(x)
        x_4 = self.conv4(x)
        x_4 = x_4.expand(-1, -1, x_1.size(2), x_1.size(3), x_1.size(4))

        # Combine parallel pathways
        x = torch.cat((x_1, x_2, x_3, x_4), dim=1)
        x = self.norm(x)
        x = self.relu(x)

        x = self.out_conv(x)

        return x
