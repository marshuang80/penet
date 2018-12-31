import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary classification.

    Adapted from:
        https://gist.github.com/AdrienLE/bf31dfe94569319f6e47b2de8df13416#file-focal_dice_1-py
    """
    def __init__(self, gamma=2, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.take_mean = size_average

    def forward(self, logits, target):
        # Inspired by the implementation of binary_cross_entropy_with_logits
        if not (target.size() == logits.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), logits.size()))

        max_val = (-logits).clamp(min=0)
        loss = logits - logits * target + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        inv_probs = F.logsigmoid(-logits * (target * 2 - 1))
        loss = (inv_probs * self.gamma).exp() * loss

        if self.take_mean:
            loss = loss.mean()

        return loss
