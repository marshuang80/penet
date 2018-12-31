import torch.nn as nn

from torchvision import models


class ResNet(nn.Module):
    """ResNet model, adapted from TorchVision to allow cams."""
    def __init__(self, model_depth, num_slices=8, device="cuda", num_classes=1, **kwargs):
        super(ResNet, self).__init__()

        model_fn = {
            18: models.resnet18,
            50: models.resnet50
        }

        self.model_depth = model_depth
        self.num_slices = num_slices
        self.device = device
        self.num_classes = num_classes

        self.model = model_fn[self.model_depth](pretrained=True)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1) 

        self.model.fc = nn.Linear(self.model.fc.in_features, 1)

    def forward(self, inputs, for_cams=False):
        num_channels = 3
        if inputs.shape[1] != num_channels:
            inputs = inputs.expand(inputs.shape[0], num_channels, inputs.shape[2], inputs.shape[3], inputs.shape[4])
        
        inputs = inputs.squeeze()
        if for_cams:
            inputs = inputs.unsqueeze(0)
        logits = self.model(inputs)
        
        return logits

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `LRCN(**model_args)`.
        """
        model_args = {'model_depth': self.model_depth,
                      'num_slices': self.num_slices,
                      'device': self.device, 
                      'num_classes': self.num_classes}

        return model_args
