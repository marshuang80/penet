import torch.nn as nn

from models.layers.penet import PENetBottleneck


class PENetEncoder(nn.Module):
    def __init__(self, in_channels, channels, num_blocks, cardinality, block_idx, total_blocks, stride=1):
        super(PENetEncoder, self).__init__()

        # Get PENet blocks
        penet_blocks = [PENetBottleneck(in_channels, channels, block_idx, total_blocks, cardinality, stride)]

        for i in range(1, num_blocks):
            penet_blocks += [PENetBottleneck(channels * PENetBottleneck.expansion, channels,
                                           block_idx + i, total_blocks, cardinality)]
        self.penet_blocks = nn.Sequential(*penet_blocks)

    def forward(self, x):
        x = self.penet_blocks(x)

        return x
