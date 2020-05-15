import os
import sys

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, MODEL_PATH)

from collections import OrderedDict
from cams import GradCAM


class CustomGradCAM(GradCAM):
    def __init__(self, model, device, is_binary, is_3d):
        super(GradCAM, self).__init__(model, device, is_binary, is_3d)
        self.fmaps = OrderedDict()
        self.grads = OrderedDict()
        self.hooks = []

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def register_hooks(self):
        def save_fmap(m, _, output):
            self.fmaps[id(m)] = output.to("cpu")

        def save_grad(m, _, grad_out):
            self.grads[id(m)] = grad_out[0].to("cpu")

        for _, module in self.model.named_modules():
            self.hooks.append(module.register_forward_hook(save_fmap))
            self.hooks.append(module.register_backward_hook(save_grad))
