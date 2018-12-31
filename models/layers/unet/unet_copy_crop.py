import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetCopyCrop(nn.Module):
    """Layer for cropping then concatenating skip connection in UNet."""
    def __init__(self):
        super(UNetCopyCrop, self).__init__()

    def forward(self, x, x_skip):
        crop_h = (x.size(2) - x_skip.size(2))
        crop_w = (x.size(3) - x_skip.size(3))
        # Round in opposite directions on either side
        x_skip = F.pad(x_skip, (crop_h//2, int(crop_h/2), crop_w//2, int(crop_w/2)))
        x = torch.cat([x_skip, x], dim=1)

        return x
