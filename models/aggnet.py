import torch.nn as nn


class AggNet(nn.Module):
    """Simple three-layer 1D convolution for aggregating model outputs."""
    def __init__(self, in_channels, **kwargs):
        super(AggNet, self).__init__()

        self.in_channels = in_channels

        self.conv1 = nn.Conv1d(in_channels, 8, kernel_size=3, bias=False)
        self.norm1 = nn.GroupNorm(2, 8)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(8, 16, kernel_size=3, bias=False)
        self.norm2 = nn.GroupNorm(4, 16)
        self.relu2 = nn.ReLU(inplace=True)

        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(16, 1)

        self._initialize_weights(init_method='kaiming')

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
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)

        x = self.max_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
