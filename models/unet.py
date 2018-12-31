import torch.nn as nn
import torch.nn.init as init
import util

from models.layers.unet import *


class UNet(nn.Module):
    """U-Net Model

    Based on the paper:
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    by Olaf Ronneberger, Philipp Fischer, Thomas Brox
    (https://arxiv.org/abs/1505.04597).
    """

    def __init__(self, model_depth=5, num_channels=1, num_classes=1, init_method=None, **kwargs):
        super(UNet, self).__init__()
        self.depth = model_depth
        self.num_channels = num_channels
        self.num_classes = num_classes

        u_channels = 64  # Number of channels at the top of the "U"
        layer_depths = [i for i in range(1, model_depth)]

        self.input_conv = UNetConvBlock(self.num_channels, u_channels)

        self.encoders = nn.ModuleList([UNetEncoder(u_channels * 2**(i - 1), u_channels * 2**i)
                                       for i in layer_depths])

        self.decoders = nn.ModuleList([UNetDecoder(u_channels * 2**i, u_channels * 2**(i - 1))
                                       for i in reversed(layer_depths)])

        self.output_conv = nn.Conv2d(u_channels, self.num_classes, 1)

        if init_method is not None:
            self._initialize_weights(init_method)

    def _initialize_weights(self, init_method):
        """Initialize all weights in the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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
            elif isinstance(m, nn.BatchNorm2d) or (isinstance(m, nn.GroupNorm) and m.affine):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_conv(x)
        skips = []

        # Encode and save skip connections
        for encoder in self.encoders:
            skips.append(x)
            x = encoder(x)

        # Decode with skip connections
        for decoder in self.decoders:
            x_skip = skips.pop()
            x = decoder(x, x_skip)

        # Generate mask
        x = self.output_conv(x)

        return x

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `LRCN(**model_args)`.
        """
        model_args = {'model_depth': self.depth,
                      'num_channels': self.num_channels,
                      'num_classes': self.num_classes}

        return model_args
