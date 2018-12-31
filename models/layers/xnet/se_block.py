from torch import nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block.

    Based on the paper:
    "Squeeze-and-Excitation Networks"
    by Jie Hu, Li Shen, Gang Sun
    (https://arxiv.org/abs/1709.01507).
    """

    def __init__(self, num_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool3d(1)
        self.excite = nn.Sequential(nn.Linear(num_channels, num_channels // reduction),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Linear(num_channels // reduction, num_channels),
                                    nn.Sigmoid())

    def forward(self, x):
        num_channels = x.size(1)

        # Squeeze
        z = self.squeeze(x)
        z = z.view(-1, num_channels)

        # Excite
        s = self.excite(z)
        s = s.view(-1, num_channels, 1, 1, 1)

        # Apply gate
        x = x * s

        return x
