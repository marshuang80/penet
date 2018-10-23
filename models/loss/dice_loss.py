import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice Loss with smoothing. Takes voxel-wise logits and labels as inputs."""
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def _dice_coefficient(probs, targets):
        """Get smoothed Dice Coefficient between `preds` and `targets`.

        Args:
            probs: Voxel-wise probabilities.
            targets: Voxel-wise binary mask of same shape as `preds`.

        Returns:
            Dice coefficient averaged over the batch dimension.

        See Also:
            https://github.com/pytorch/pytorch/issues/1249
        """
        smooth = 1.

        batch_size = probs.size(0)
        m1 = probs.view(batch_size, -1)
        m2 = targets.view(batch_size, -1).type(torch.float32)

        intersection = m1 * m2
        m1_sq = m1 * m1
        m2_sq = m2 * m2

        dice = 2. * (intersection.sum(1) + smooth) / (m1_sq.sum(1) + m2_sq.sum(1) + smooth)
        dice = dice.mean()

        return dice

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        loss = 1. - self._dice_coefficient(probs, targets)
        return loss
