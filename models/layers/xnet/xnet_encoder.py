import torch.nn as nn

from models.layers.xnet import XNetBottleneck


class XNetEncoder(nn.Module):
    def __init__(self, in_channels, channels, num_blocks, cardinality, block_idx, total_blocks, stride=1):
        super(XNetEncoder, self).__init__()

        # Get XNet blocks
        xnet_blocks = [XNetBottleneck(in_channels, channels, block_idx, total_blocks, cardinality, stride)]

        for i in range(1, num_blocks):
            xnet_blocks += [XNetBottleneck(channels * XNetBottleneck.expansion, channels,
                                           block_idx + i, total_blocks, cardinality)]
        self.xnet_blocks = nn.Sequential(*xnet_blocks)

    def forward(self, x):
        x = self.xnet_blocks(x)

        return x
