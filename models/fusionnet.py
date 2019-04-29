from saver import ModelSaver
import torch.nn as nn
import torch
from models.layers.xnet import *


class FusionNet(nn.Module):
    """ResNet model, adapted from TorchVision to allow cams."""
    def __init__(self, args, num_meta):
        super(FusionNet, self).__init__()

        # load pretrained vision model
        model, ckpt_info = ModelSaver.load_model(args.ckpt_path, args.gpu_ids)


        # freeze weights
        for param in model.parameters():
            param.requires_grad = False

        
        # print number of trainable/frozen weights
        #total_params = sum(p.numel() for p in model.parameters())
        #trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        #print("Total Params: ", total_params)
        #print("Trainable Params: ", trainable_params)

        model.module.classifier = Identity()
        self.pretrained_model = model
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.batch_norm = nn.BatchNorm1d(2048 + num_meta)
        self.classifier = nn.Linear(2048 + num_meta, 1)
        #self.classifier = nn.Sequential(nn.Linear(2048 + num_meta, 512),
		#							    nn.Linear(512, 512),
		#							    nn.Linear(512, 1))



        self.classifier.is_output_head = True


    def forward(self, inputs, meta):

        # TODO check if expand is nessasary (check Xnet)
        #meta = torch.FloatTensor(meta)

        x = self.pretrained_model(inputs)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)

        fused = torch.cat((x, meta), 1)
        fused = fused.view(fused.size(0), -1)
        #fused = self.batch_norm(fused)

        pred = self.classifier(fused)

        return pred

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
