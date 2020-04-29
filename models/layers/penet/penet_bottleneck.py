import random
import torch.nn as nn

from models.layers.penet import SEBlock


class PENetBottleneck(nn.Module):
    """PENet bottleneck block, similar to a pre-activation ResNeXt bottleneck block.

    Based on the paper:
    "Aggregated Residual Transformations for Deep Nerual Networks"
    by Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He
    (https://arxiv.org/abs/1611.05431).
    """

    expansion = 2

    def __init__(self, in_channels, channels, block_idx, total_blocks, cardinality=32, stride=1):
        super(PENetBottleneck, self).__init__()
        mid_channels = cardinality * int(channels / cardinality)
        out_channels = channels * self.expansion
        self.survival_prob = self._get_survival_prob(block_idx, total_blocks)

        self.down_sample = None
        if stride != 1 or in_channels != channels * PENetBottleneck.expansion:
            self.down_sample = nn.Sequential(
                nn.Conv3d(in_channels, channels * PENetBottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(channels * PENetBottleneck.expansion // 16, channels * PENetBottleneck.expansion))

        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.norm1 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu1 = nn.LeakyReLU(inplace=True)

        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3,
                               stride=stride, padding=1, groups=cardinality, bias=False)
        self.norm2 = nn.GroupNorm(mid_channels // 16, mid_channels)
        self.relu2 = nn.LeakyReLU(inplace=True)

        self.conv3 = nn.Conv3d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.norm3 = nn.GroupNorm(out_channels // 16, out_channels)
        self.norm3.is_last_norm = True
        self.relu3 = nn.LeakyReLU(inplace=True)

        self.se_block = SEBlock(out_channels, reduction=16)

    @staticmethod
    def _get_survival_prob(block_idx, total_blocks, p_final=0.5):
        """Get survival probability for stochastic depth. Uses linearly decreasing
        survival probability as described in "Deep Networks with Stochastic Depth".

        Args:
            block_idx: Index of residual block within entire network.
            total_blocks: Total number of residual blocks in entire network.
            p_final: Survival probability of the final layer.
        """
        return 1. - block_idx / total_blocks * (1. - p_final)

    def forward(self, x):
        x_skip = x if self.down_sample is None else self.down_sample(x)

        # Stochastic depth dropout
        if self.training and random.random() > self.survival_prob:
            return x_skip

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)

        x = self.se_block(x)
        x += x_skip

        x = self.relu3(x)

        return x
