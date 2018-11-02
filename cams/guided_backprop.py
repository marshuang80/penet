from cams import BaseCAM
import torch
import torch.nn as nn
import torch.nn.functional as F

class GuidedBackPropagation(BaseCAM):

    def __init__(self, model, device, is_binary, is_3d):
        super(GuidedBackPropagation, self).__init__(model, device, is_binary, is_3d)
        self.input_grad = []
        def func_b(module, grad_in, grad_out):
            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)


    def generate(self):
        output = self.input_grad.to('cpu').numpy()[0]
        return output

    def forward(self, x):
        self.inputs = x.to(self.device)

        def save_grad(grad):
            self.input_grad = grad.to('cpu')

        self.inputs.register_hook(save_grad)
        self.model.zero_grad()
        self.preds = self.model(self.inputs)

        if self.is_binary:
            self.probs = F.sigmoid(self.preds)[0]
        else:
            self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.sort(0, True)

        return self.prob, self.idx