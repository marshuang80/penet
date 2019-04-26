from saver import ModelSaver
import torch.nn as nn
import torch
from models.layers.xnet import *


class FusionNet(nn.Module):
    """ResNet model, adapted from TorchVision to allow cams."""
    def __init__(self, args, num_meta):
        super(FusionNet, self).__init__()
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)

        
        #print(model)
        # remove last layer
        #model = torch.nn.Sequential(*(list(model.children())))
        model.module.classifier = Identity()

        # freeze model weighhs
        for param in model.parameters():
            param.require_gard = False

        self.pretrained_model = model
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(2048 + num_meta, 1)
        self.classifier.is_output_head = True


    def forward(self, inputs, meta):

        # TODO check if expand is nessasary (check Xnet)
        #meta = torch.FloatTensor(meta)

        x = self.pretrained_model(inputs)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)

        fused = torch.cat((x, meta), 1)
        fused = fused.view(fused.size(0), -1)

        pred = self.classifier(fused)

        return pred

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
