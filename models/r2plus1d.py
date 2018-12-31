import torch.nn as nn
import torch.nn.init as init

from models.layers.r2plus1d import *


class R2Plus1D(nn.Module):
    """R(2+1)D Model.

    Essentially a 3D ResNet, but with 3D convolution replaced by (2+1)D convolution.
    Input must have shape (c, 8*t, 16*h, 16*w) for positive integers c, t, h, w.

    Based on the paper:
    "A Closer Look at Spatiotemporal Convolutions for Action Recognition"
    by Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, Manohar Paluri
    (https://arxiv.org/abs/1711.11248).
    """
    def __init__(self, model_depth=18, num_channels=1, num_labels=1, init_method=None, **kwargs):
        """
        Args:
            model_depth: Total number of ResNet-style basic blocks.
            num_channels: Number of channels in the input.
            num_labels: Number of labels for the output.
            init_method: Method for initializing parameters ('kaiming', 'xavier', 'normal').
        """
        super(R2Plus1D, self).__init__()

        self.model_depth = model_depth
        self.num_channels = num_channels
        self.num_labels = num_labels

        self.features = self._make_layers(self.model_depth, self.num_channels)
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(512, self.num_labels)
        if init_method is not None:
            self._initialize_weights(init_method)

    @staticmethod
    def _make_layers(model_depth, in_channels, use_bias=True, bn_eps=1e-3, bn_momentum=0.1):
        """Make layers for R(2+1)D, up through the Global Average Pooling layer."""
        block_config = {
            10: (1, 1, 1, 1),
            16: (2, 2, 2, 1),
            18: (2, 2, 2, 2),
            26: (2, 3, 4, 3),
            34: (3, 4, 6, 3),
        }
        n1, n2, n3, n4 = block_config[model_depth]

        layers = [nn.Conv3d(in_channels, out_channels=45, kernel_size=(1, 7, 7),
                            stride=(1, 2, 2), padding=(0, 3, 3), bias=use_bias),
                  nn.BatchNorm3d(45, eps=bn_eps, momentum=bn_momentum),
                  nn.ReLU(inplace=True),
                  nn.Conv3d(45, out_channels=64, kernel_size=(3, 1, 1),
                            stride=(1, 1, 1), padding=(1, 0, 0), bias=use_bias),
                  nn.BatchNorm3d(64, eps=bn_eps, momentum=bn_momentum),
                  nn.ReLU(inplace=True)]

        layers += [R2Plus1DBlock(64, 64) for _ in range(n1)]

        layers += [R2Plus1DBlock(64, 128, down_sample=True)]
        layers += [R2Plus1DBlock(128, 128) for _ in range(n2 - 1)]

        layers += [R2Plus1DBlock(128, 256, down_sample=True)]
        layers += [R2Plus1DBlock(256, 256) for _ in range(n3 - 1)]

        layers += [R2Plus1DBlock(256, 512, down_sample=True)]
        layers += [R2Plus1DBlock(512, 512) for _ in range(n4 - 1)]

        features = nn.Sequential(*layers)

        return features

    def _initialize_weights(self, init_method):
        """Initialize all weights in the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
                if init_method == 'normal':
                    init.normal_(m.weight, 0.0, 0.2)
                elif init_method == 'xavier':
                    init.xavier_normal_(m.weight, 0.2)
                elif init_method == 'kaiming':
                    init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                else:
                    raise NotImplementedError('Invalid initialization method: {}'.format(self.init_method))
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.

        To use the returned dict, initialize the model with `R2Plus1D(**model_args)`.
        """
        model_args = {'model_depth': self.model_depth,
                      'num_channels': self.num_channels,
                      'num_labels': self.num_labels}

        return model_args
