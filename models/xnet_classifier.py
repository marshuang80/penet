import math
import torch.nn as nn

from models.layers.xnet import *


class XNetClassifier(nn.Module):
    """XNet stripped down for classification.

    The idea is to pre-train this network, then use the pre-trained
    weights for the encoder in a full XNet.
    """

    def __init__(self, model_depth, cardinality=32, num_channels=3, num_classes=600, init_method=None, **kwargs):
        super(XNetClassifier, self).__init__()

        self.in_channels = 64
        self.model_depth = model_depth
        self.cardinality = cardinality
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.in_conv = nn.Sequential(nn.Conv3d(self.num_channels, self.in_channels, kernel_size=7,
                                               stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
                                     nn.GroupNorm(self.in_channels // 16, self.in_channels),
                                     nn.LeakyReLU(inplace=True))
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # Encoders
        if model_depth != 50:
            raise ValueError('Unsupported model depth: {}'.format(model_depth))
        encoder_config = [3, 4, 6, 3]
        total_blocks = sum(encoder_config)
        block_idx = 0

        self.encoders = nn.ModuleList()
        for i, num_blocks in enumerate(encoder_config):
            out_channels = 2 ** i * 128
            stride = 1 if i == 0 else 2
            encoder = XNetEncoder(self.in_channels, out_channels, num_blocks, self.cardinality,
                                  block_idx, total_blocks, stride=stride)
            self.encoders.append(encoder)
            self.in_channels = out_channels * XNetBottleneck.expansion
            block_idx += num_blocks

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(self.in_channels, num_classes)

        if init_method is not None:
            self._initialize_weights(init_method)

    def _initialize_weights(self, init_method, gain=0.2):
        """Initialize all weights in the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear):
                if init_method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                else:
                    raise NotImplementedError('Invalid initialization method: {}'.format(self.init_method))
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm) and m.affine:
                # Gamma for last GroupNorm in each residual block gets set to 0
                init_gamma = 0 if hasattr(m, 'is_last_norm') and m.is_last_norm else 1
                nn.init.constant_(m.weight, init_gamma)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # Expand input (allows pre-training on RGB videos, fine-tuning on Hounsfield Units)
        if x.size(1) < self.num_channels:
            x = x.expand(-1, self.num_channels // x.size(1), -1, -1, -1)

        x = self.in_conv(x)
        x = self.max_pool(x)

        # Encoders
        for encoder in self.encoders:
            x = encoder(x)

        # Classifier
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `XNet(**model_args)`.
        """
        model_args = {
            'model_depth': self.model_depth,
            'cardinality': self.cardinality,
            'num_classes': self.num_classes,
            'num_channels': self.num_channels
        }

        return model_args
