import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from submodules.model_utils import init_model
from submodules import GAPLinear, R2Plus1DBlock

# Maps model depth -> Block size per layer
BLOCK_CONFIG = {
    10: (1, 1, 1, 1),
    16: (2, 2, 2, 1),
    18: (2, 2, 2, 2),
    26: (2, 3, 4, 3),
    34: (3, 4, 6, 3),
}

class R2Plus1DModel(nn.Module):
    """R(2+1)D Model.
    If input shape is (c, 8*t, 16*h, 16*w),
    then final conv output shape is (512, t, h, w).
    Based on the paper:
    "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    by Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri
    (https://arxiv.org/abs/1711.11248).
    """
    def __init__(self, config, num_classes):
        super().__init__()
        config.model_depth = 10
        self.model = self._build_model(config.model_depth, config.num_channels, num_classes)
        self.model = init_model(self.model, 'xavier')

    @staticmethod
    def _build_model(model_depth, num_input_channels, num_classes,
                     use_bias=True, bn_eps=1e-3, bn_momentum=0.9):

        n1, n2, n3, n4 = BLOCK_CONFIG[model_depth]

        layers = [nn.Conv3d(num_input_channels, out_channels=45, kernel_size=(1, 7, 7),
                            stride=(1, 2, 2), padding=(0, 3, 3), bias=use_bias),
                  nn.BatchNorm3d(45, eps=bn_eps, momentum=bn_momentum),
                  nn.ReLU(),
                  nn.Conv3d(45, out_channels=64, kernel_size=(3, 1, 1),
                            stride=(1, 1, 1), padding=(1, 0, 0), bias=use_bias),
                  nn.BatchNorm3d(64, eps=bn_eps, momentum=bn_momentum),
                  nn.ReLU()]

        layers += [R2Plus1DBlock(64, 64) for _ in range(n1)]

        layers += [R2Plus1DBlock(64, 128, down_sample=True)]
        layers += [R2Plus1DBlock(128, 128) for _ in range(n2 - 1)]

        layers += [R2Plus1DBlock(128, 256, down_sample=True)]
        layers += [R2Plus1DBlock(256, 256) for _ in range(n3 - 1)]

        layers += [R2Plus1DBlock(256, 512, down_sample=True)]
        layers += [R2Plus1DBlock(512, 512) for _ in range(n4 - 1)]
        layers += [GAPLinear(512, num_classes)]

        model = nn.Sequential(*layers)
        return model

    def forward(self, X):
        # _input is batch_size, n_images, n_channels, h, w
        # Needs to be batch_size, n_channels, n_images, h, w
        X = X.permute((0, 2, 1, 3, 4))
        out = self.model(X)
        return out
