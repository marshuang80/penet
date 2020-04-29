import torch
import torch.nn.functional as F

from collections import OrderedDict
from .base_cam import BaseCAM


class GradCAM(BaseCAM):
    """Class for generating grad CAMs.

    Adapted from: https://github.com/kazuto1011/grad-cam-pytorch
    """
    def __init__(self, model, device, is_binary, is_3d):
        super(GradCAM, self).__init__(model, device, is_binary, is_3d)
        self.fmaps = OrderedDict()
        self.grads = OrderedDict()

        def save_fmap(m, _, output):
            self.fmaps[id(m)] = output.to('cpu')

        def save_grad(m, _, grad_out):
            self.grads[id(m)] = grad_out[0].to('cpu')

        for _, module in self.model.named_modules():
            module.register_forward_hook(save_fmap)
            module.register_backward_hook(save_grad)

    def _find(self, outputs, target_layer):
        for k, v in outputs.items():
            for name, module in self.model.named_modules():
                if id(module) == k:
                    if name == target_layer:
                        return v
        raise ValueError('Invalid layer name: {}'.format(target_layer))

    @staticmethod
    def _normalize(grads):
        return grads / (torch.norm(grads).item() + 1e-5)

    def _compute_grad_weights(self, grads):
        grads = self._normalize(grads)
        if self.is_3d:
            weights = F.adaptive_avg_pool3d(grads, 1)
        else:
            weights = F.adaptive_avg_pool2d(grads, 1)
        return weights

    def get_cam(self, target_layer):
        fmaps = self._find(self.fmaps, target_layer)
        grads = self._find(self.grads, target_layer)
        weights = self._compute_grad_weights(grads)

        gcam = (fmaps[0] * weights[0]).sum(dim=0)

        gcam -= gcam.min()
        gcam /= gcam.max()

        print(gcam)

        return gcam.detach().to('cpu').numpy()
