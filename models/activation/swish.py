import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish activation function

    Based on the paper:
    "Searching for Activation Functions"
    by Prajit Ramachandran, Barret Zoph, Quoc V. Le
    (https://arxiv.org/abs/1710.05941).
    """
    def __init__(self, trainable=False):
        super(Swish, self).__init__()
        self.beta = None
        if trainable:
            self.beta = torch.ones([], dtype=torch.float32, requires_grad=True)

    def forward(self, x):
        if self.beta is None:
            # Not trainable
            x = x * x.sigmoid()
        else:
            # Trainable
            x = x * torch.sigmoid(self.beta * x)

        return x
