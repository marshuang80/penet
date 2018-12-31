import torch
import torch.nn as nn
import torch.nn.init as init

from .layers.vnet import VNetConvBlock, VNetDecoder, VNetEncoder


class VNet(nn.Module):
    """V-Net model for 3D segmentation

    For model_depth of 5, input and output tensors have shape (batch_size, 1, 64, 128, 128).

    Based on the paper:
    "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
    by Fausto Milletari, Nassir Navab, Seyed-Ahmad Ahmadi
    (https://arxiv.org/abs/1606.04797).
    """
    def __init__(self, model_depth=5, num_channels=1, num_classes=1,
                 dropout_prob=0., init_method=None, **kwargs):
        super(VNet, self).__init__()
        self.dropout_prob = dropout_prob
        self.model_depth = model_depth
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.v_channels = 16  # Number of channels at the top of the "V"

        self.input_block = VNetConvBlock(1, self.v_channels)

        # Left-hand side of the "V"
        layer_depths = [i for i in range(1, model_depth)]
        self.encoders = nn.ModuleList()
        for layer_depth in layer_depths:
            in_channels, out_channels, num_layers, dropout_prob = self._get_layer_args(layer_depth, is_encoder=True)
            self.encoders.append(VNetEncoder(in_channels, out_channels, num_layers, dropout_prob))

        # Right-hand side of the "V"
        self.decoders = nn.ModuleList()
        for layer_depth in reversed(layer_depths):
            in_channels, out_channels, num_layers, dropout_prob = self._get_layer_args(layer_depth, is_encoder=False)
            self.decoders.append(VNetDecoder(in_channels, out_channels, num_layers, dropout_prob))

        self.output_conv = nn.Conv3d(2 * self.v_channels, self.num_classes, kernel_size=1)

        if init_method is not None:
            self._initialize_weights(init_method)

    def _get_layer_args(self, layer_depth, is_encoder):
        """Get args for a VNetEncoder or VNetDecoder at the specified layer depth."""
        out_channels = self.v_channels * 2 ** layer_depth
        if is_encoder:
            in_channels = out_channels // 2
        else:
            # Bottom decoder has same number of input and output channels
            in_channels = out_channels * (2 if layer_depth < self.model_depth - 1 else 1)

        if layer_depth == 0:
            # In the paper, final decoder just has a single conv layer
            num_layers = 2 if is_encoder else 1
        else:
            num_layers = 3

        # Only use dropout beneath the second layer
        use_dropout = 0.0 if layer_depth <= 2 else self.dropout_prob

        return in_channels, out_channels, num_layers, use_dropout

    def _initialize_weights(self, init_method):
        """Initialize all weights in the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                if init_method == 'normal':
                    init.normal_(m.weight, 0.0, 0.2)
                elif init_method == 'xavier':
                    init.xavier_normal_(m.weight, 0.2)
                elif init_method == 'kaiming':
                    init.kaiming_normal_(m.weight)
                else:
                    raise NotImplementedError('Invalid initialization method: {}'.format(self.init_method))
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or (isinstance(m, nn.GroupNorm) and m.affine):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.input_block(x)

        # Encode and save skip connections
        skips = []
        for encoder in self.encoders:
            skips.append(x)
            x = encoder(x)

        # Decode with skip connections
        for decoder in self.decoders:
            x_skip = skips.pop()
            x = decoder(x, x_skip)

        # Generate mask
        x = self.output_conv(x)
        x = torch.squeeze(x, 1)

        return x

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `VNet(**model_args)`.
        """
        model_args = {'model_depth': self.model_depth,
                      'num_channels': self.num_channels,
                      'num_classes': self.num_classes,
                      'dropout_prob': self.dropout_prob}

        return model_args
