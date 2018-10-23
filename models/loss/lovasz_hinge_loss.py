import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import filterfalse


class LovaszHingeLoss(nn.Module):
    """Binary Lovasz hinge loss.

    Args:
        per_image: compute the loss per image instead of per batch.
        ignore: Class id to ignore

    Adapted from:
        https://github.com/bermanmaxim/LovaszSoftmax/
    """

    def __init__(self, per_image=True, ignore=None):
        super(LovaszHingeLoss, self).__init__()
        self.per_image = per_image
        self.ignore = ignore

    @staticmethod
    def _mean(lst, ignore_nan=False, empty=0):
        """NaN-mean compatible with generators."""
        lst = iter(lst)
        if ignore_nan:
            lst = filterfalse(np.isnan, lst)
        try:
            n = 1
            acc = next(lst)
        except StopIteration:
            if empty == 'raise':
                raise ValueError('Empty mean')
            return empty
        for n, v in enumerate(lst, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def _lovasz_hinge_flat(self, logits, labels):
        """Binary Lovasz hinge loss

        Args:
            logits: Logits at each prediction (between -\infty and +\infty).
            labels: Tensor, binary ground truth labels (0 or 1).
        """
        if len(labels) == 0:
            # only void pixels, the gradients should be 0
            return logits.sum() * 0.
        signs = 2. * labels - 1.
        errors = (1. - logits * signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.relu(errors_sorted), grad)
        return loss

    @staticmethod
    def _flatten_binary_scores(scores, labels, ignore=None):
        """Flattens predictions in the batch (binary case).
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = (labels != ignore)
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    def forward(self, logits, labels):
        if self.per_image:
            loss = self._mean(
                self._lovasz_hinge_flat(*self._flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), self.ignore))
                for log, lab in zip(logits, labels))
        else:
            loss = self._lovasz_hinge_flat(*self._flatten_binary_scores(logits, labels, self.ignore))

        return loss
