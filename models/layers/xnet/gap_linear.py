import torch.nn as nn


class GAPLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        """Global average pooling (3D) followed by a linear layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels
        """
        super(GAPLinear, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, out_channels)
        self.fc.is_output_head = True

    def forward(self, x):
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
