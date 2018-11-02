import torch
import torch.nn.functional as F


class BaseCAM(object):
    """Base class for generating CAMs.

    Adapted from: https://github.com/kazuto1011/grad-cam-pytorch
    """
    def __init__(self, model, device, is_binary, is_3d):
        super(BaseCAM, self).__init__()
        self.device = device
        self.is_binary = is_binary
        self.is_3d = is_3d
        self.model = model
        self.model.eval()
        self.inputs = None

    def _encode_one_hot(self, idx):
        one_hot = torch.zeros([1, self.preds.size()[-1]],
                              dtype=torch.float32, device=self.device, requires_grad=True)
        one_hot[0][idx] = 1.0

        return one_hot

    def forward(self, x):
        self.inputs = x.to(self.device)
        self.model.zero_grad()
        self.preds = self.model(self.inputs)

        if self.is_binary:
            self.probs = F.sigmoid(self.preds)[0]
        else:
            self.probs = F.softmax(self.preds, dim=1)[0]
        self.prob, self.idx = self.probs.sort(0, True)

        return self.prob, self.idx

    def backward(self, idx):
        one_hot = self._encode_one_hot(idx)
        self.preds.backward(gradient=one_hot, retain_graph=True)

    def get_cam(self, target_layer):
        raise NotImplementedError
