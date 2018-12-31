import torch.nn as nn


class GAPLinear(nn.Module):
    def __init__(self, in_features, out_features):
        """Global Average Pooling followed by linear output layer.
        Args:
            in_features: Number of input features to the linear layer.
            out_features: Number of output features from the linear layer.
        """
        super(GAPLinear, self).__init__()
        self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, input_):
        # GAP, reshape, linear
        avg_pool = self.gap(input_)
        avg_pool = avg_pool.view(avg_pool.size(0), avg_pool.size(1))
        return self.linear(avg_pool)
