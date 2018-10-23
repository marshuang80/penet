"""
model.py
    Defines PyTorch nn.Module classes.
    Each should implement a constructor __init__(self, config)
    and a forward pass forward(self, x)
"""
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torchvision import models
import torch.nn.functional as F
import math
import torch.autograd as autograd
import psutil
from collections import OrderedDict

import os

class ResNet18(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.resnet18(pretrained=config.pretrained)
        num_ftrs = self.model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.config = config

    def get_features(self, X):
        x = self.model.conv1(X)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, X):
        # Expect input of size [Batch_size, #Images, #Channels, H, W]
        batch_size, n_channels, h, w = X.size()
        features = self.get_features(X)
        features = features.permute((0, 2, 1))
        out = self.fc(features)
        return out


class DenseNet121(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.densenet121(pretrained=config.pretrained)
        self.num_ftrs = self.model.classifier.in_features
        
        self.fc = nn.Linear(self.num_ftrs, num_classes)
        self.dropout = nn.Dropout2d(p=0.3)

    def get_features(self, X):
        x = self.model.features(X)
        x = nn.functional.relu(x, inplace=True)
        return x

    def forward(self, X):
        # Expect input of size [Batch_size, #Images, #Channels, H, W]
        batch_size, n_channels, h, w = X.size()
        features = self.get_features(X)
        features = features.permute((0, 2, 1))
        out = self.fc(features)
        return out


class AlexNet(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.model = models.alexnet(pretrained=config.pretrained)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, X):
        batch_size, n_channels, h, w = X.size()
        features = self.get_features(X)
        features = features.permute((0, 2, 1))
        out = self.fc(features)
        return out


class VGG13(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.vgg13(pretrained=config.pretrained)
        self.fc = nn.Linear(4096, num_classes)
        self.model.classifier = nn.Linear(25088, 4096)
    def get_features(self, X):
        x = self.model.features(X)
        x = x.view(x.size(0), 25088)
        x = nn.functional.relu(self.model.classifier(x))
        x = self.fc(x)
        return x


    def forward(self, X):
        # Expect input of size [Batch_size, #Images, #Channels, H, W]
        batch_size, n_channels, h, w = X.size()
        features = self.get_features(X)
        features = features.permute((0, 2, 1))
        out = self.fc(features)
        return out


class VGG16(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.vgg16(pretrained=config.pretrained)
        num_ftrs = 4096 #self.model.fc.in_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.config = config

    def get_features(self, X):
        x = self.model.features[0](X)
        x = nn.functional.relu(x)
        x = self.model.features[2](x)
        x = nn.functional.relu(x)
        x = self.model.features[4](x)
        x = self.model.features[5](x)
        x = nn.functional.relu(x)
        x = self.model.features[7](x)
        x = nn.functional.relu(x)
        x = self.model.features[9](x)
        x = self.model.features[10](x)
        x = nn.functional.relu(x)
        x = self.model.features[12](x)
        x = nn.functional.relu(x)
        x = self.model.features[14](x)
        x = nn.functional.relu(x)
        x = self.model.features[16](x)
        x = self.model.features[17](x)
        x = nn.functional.relu(x)
        x = self.model.features[19](x)
        x = nn.functional.relu(x)
        x = self.model.features[21](x)
        x = nn.functional.relu(x)
        x = self.model.features[23](x)
        x = self.model.features[24](x)
        x = nn.functional.relu(x)
        x = self.model.features[26](x)
        x = nn.functional.relu(x)
        x = self.model.features[28](x)
        x = nn.functional.relu(x)
        x = self.model.features[30](x)

        x = x.view(x.size(0), -1)
        x = self.model.classifier[0](x) # (0): Linear(in_features=25088, out_features=4096)
        x = nn.functional.relu(x)       # (1): ReLU(inplace)
        x = self.model.classifier[2](x) # (2): Dropout(p=0.5)
        x = self.model.classifier[3](x) # (3): Linear(in_features=4096, out_features=4096)
        x = nn.functional.relu(x)       # (4): ReLU(inplace)
        x = self.model.classifier[5](x) # (5): Dropout(p=0.5)
        return x

    def forward(self, X):
        # Expect input of size [Batch_size, #Images, #Channels, H, W]
        batch_size, n_channels, h, w = X.size()
        features = self.get_features(X)
        out = self.fc(features)
        return out



model_dict = {'SeqAvgDenseNet121': SeqAvgDenseNet121,
              'SeqAvgResNet18': SeqAvgResNet18,
              'SeqAvgAlexNet': SeqAvgAlexNet,
              'SeqAvgVGG13': SeqAvgVGG13,
              'SeqAvgVGG16': SeqAvgVGG16,
              'DoublePoolAlexNet':DoublePoolAlexNet} 


