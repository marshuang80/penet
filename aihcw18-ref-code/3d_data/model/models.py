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

#from r2plus1d import R2Plus1DModel
#import resnet3D
#import resnext3D
#import wide_resnet3D
#import densenet3D
import os

'''
class ThreeDDenseNet121(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Only ThreeDResNeXt101 has a pretrained model. The others have been ignored 
        # in the model_dict

        # Naming this self.module is important for the state_dict to load properly
        self.module = densenet3D.densenet121(num_classes=num_classes,
                sample_size=config.batch_size, sample_duration=1)


    def forward(self, X):
        X = X.permute(0, 2, 1, 3, 4)
        out = self.module.forward(X)
        return out

# This gives an unfortunate error: library not initialized
class ThreeDWideResNet50(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Only ThreeDResNeXt101 has a pretrained model. The others have been ignored 
        # in the model_dict

        # Naming this self.module is important for the state_dict to load properly
        self.module = wide_resnet3D.resnet50(num_classes=num_classes, k=2,
                shortcut_type='B', sample_size=config.batch_size, sample_duration=1)


    def forward(self, X):
        X = X.permute(0, 2, 1, 3, 4)
        out = self.module.forward(X)
        return out


class ThreeDResNeXt(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Only ThreeDResNeXt101 has a pretrained model. The others have been ignored 
        # in the model_dict

        # Naming this self.module is important for the state_dict to load properly
        if config.model == 'ThreeDResNeXt50':
            self.module = resnext3D.resnet50(num_classes=num_classes,
                    shortcut_type='B', sample_size=config.batch_size, sample_duration=1)
        if config.model == 'ThreeDResNeXt101':
            self.module = resnext3D.resnet101(num_classes=num_classes,
                    shortcut_type='B', sample_size=config.batch_size, sample_duration=1)
        if config.model == 'ThreeDResNeXt152':
            self.module = resnext3D.resnet152(num_classes=num_classes,
                    shortcut_type='B', sample_size=config.batch_size, sample_duration=1)


    def forward(self, X):
        X = X.permute(0, 2, 1, 3, 4)
        out = self.module.forward(X)
        return out



class ThreeDResNet(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        # Naming this self.module is important for the state_dict to load properly
        if config.model == 'ThreeDResNet10':
            self.module = resnet3D.resnet10(num_classes=num_classes,
                    shortcut_type='A', sample_size=config.batch_size, sample_duration=1)
        if config.model == 'ThreeDResNet18':
            self.module = resnet3D.resnet18(num_classes=num_classes,
                    shortcut_type='A', sample_size=config.batch_size, sample_duration=1)
        if config.model == 'ThreeDResNet34':
            self.module = resnet3D.resnet34(num_classes=num_classes,
                    shortcut_type='A', sample_size=config.batch_size, sample_duration=1)
        if config.model == 'ThreeDResNet50':
            self.module = resnet3D.resnet50(num_classes=num_classes,
                    shortcut_type='B', sample_size=config.batch_size, sample_duration=1)
        if config.model == 'ThreeDResNet101':
            self.module = resnet3D.resnet101(num_classes=num_classes,
                    shortcut_type='B', sample_size=config.batch_size, sample_duration=1)
        if config.model == 'ThreeDResNet152':
            self.module = resnet3D.resnet152(num_classes=num_classes,
                    shortcut_type='B', sample_size=config.batch_size, sample_duration=1)
        if config.model == 'ThreeDResNet200':
            self.module = resnet3D.resnet200(num_classes=num_classes,
                    shortcut_type='B', sample_size=config.batch_size, sample_duration=1)


    def forward(self, X):
        X = X.permute(0, 2, 1, 3, 4)
        out = self.module.forward(X)
        return out
'''

class SeqAvgResNet18(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.resnet18(pretrained=config.pretrained)
        self.adapool = nn.AdaptiveAvgPool2d(1)
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
        x = self.adapool(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, X):
        # Expect input of size [Batch_size, #Images, #Channels, H, W]
        batch_size, n_images, n_channels, h, w = X.size()
        X = X.view(-1, 3, h, w)
        features = self.get_features(X)
        features = features.view(batch_size, n_images, -1)
        features = features.permute((0, 2, 1))
        features_pooled = torch.nn.functional.adaptive_avg_pool1d(features, 1).view(batch_size, -1)
        out = self.fc(features_pooled)
        return out

class SeqAvgAlexNet(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.alexnet(pretrained=config.pretrained)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.config = config
        self.fc = nn.Linear(9216, num_classes)
        #self.model.classifier = nn.Linear(256, 4096)
        self.dropout = nn.Dropout()


    def get_features(self, X):
        x = self.model.features(X)
        x = x.view(x.size(0), -1)
        return x


    def forward(self, X):
        # Expect input of size [Batch_size, #Images, #Channels, H, W]
        batch_size, n_images, n_channels, h, w = X.size()
        X = X.view(-1, 3, h, w)
        features = self.get_features(X)
        features = features.view(batch_size, n_images, -1)
        features = features.permute((0, 2, 1))
        features = torch.nn.functional.adaptive_avg_pool1d(features, 1).view(batch_size, -1)
        out = self.fc(features)
        return out

# This is Max-Avg
class SeqAvgDenseNet121(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.densenet121(pretrained=config.pretrained)
        self.num_ftrs = self.model.classifier.in_features
        
        self.pooling = config.pooling
        self.fc = nn.Linear(self.num_ftrs, num_classes)
        self.config = config
        self.dropout = nn.Dropout2d(p=0.3)

    def get_features(self, X):
        x = self.model.features(X)
        x = nn.functional.relu(x, inplace=True)
        # x = self.adapool(x)
        return x

    def forward(self, X):
        # Expect input of size [Batch_size, #Images, #Channels, H, W]
        batch_size, n_images, n_channels, h, w = X.size()
        X = X.view(-1, 3, h, w)
        features = self.get_features(X)
        features = features.view(batch_size *  n_images, self.num_ftrs, -1)
        features = torch.nn.functional.adaptive_max_pool1d(features, 1)
        features = features.view(batch_size, n_images, self.num_ftrs)
        features = features.permute((0, 2, 1))
        features = torch.nn.functional.adaptive_avg_pool1d(features, 1).view(batch_size, -1)
        out = self.fc(features)
        return out


class DoublePoolAlexNet(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.pooling = config.pooling
        assert self.pooling in ['avgavg','avgmax','maxavg','maxmax'], 'Invalid pooling argument'
        self.model = models.alexnet(pretrained=config.pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1) if self.pooling in ['avgavg','avgmax'] else nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, X):
        batch_size, num_slices, depth, rows, cols = X.size()
        if batch_size != 1:
            raise ValueError('Only implemented for batch size 1')
        X = torch.squeeze(X, dim=0)
        features = self.model.features(X)
        
        if self.pooling in ['avgavg','maxavg']:
            out = self.pool(features).view(features.size(0), -1)
            out = torch.mean(out, 0, keepdim=True)
        else:
            out = self.pool(features).view(features.size(0), -1)
            out = torch.max(out, 0, keepdim=True)[0]

        out = self.classifier(out)
        return out


class SeqAvgVGG13(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.vgg13(pretrained=config.pretrained)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4096, num_classes)
        self.model.classifier = nn.Linear(25088, 4096)
    def get_features(self, X):
        x = self.model.features(X)
        x = x.view(x.size(0), 25088)
        x = self.model.classifier(x)
        return x


    def forward(self, X):
        # Expect input of size [Batch_size, #Images, #Channels, H, W]
        batch_size, n_images, n_channels, h, w = X.size()
        X = X.view(-1, 3, h, w)
        features = self.get_features(X)
        features = features.view(batch_size, n_images, -1)
        features = features.permute((0, 2, 1))
        features = torch.nn.functional.adaptive_avg_pool1d(features, 1).view(batch_size, -1)
        out = self.fc(features)
        return out


class SeqAvgVGG16(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.vgg16(pretrained=config.pretrained)
        # self.adapool = nn.AdaptiveAvgPool2d(1)
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
        batch_size, n_images, n_channels, h, w = X.size()
        X = X.view(-1,3,h,w)
        features = self.get_features(X)
        # features = features.view(batch_size, n_images, -1) # Only relevant if batch_size > 1
        features_pooled = torch.mean(features, dim=0, keepdim=True)
        out = self.fc(features_pooled)
        return out



class C3D(nn.Module):
    def __init__(self, args, num_classes):
        super().__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        #self.fc6 = nn.Linear(8192, 4096)
        #self.fc7 = nn.Linear(4096, 4096)
        #self.fc8 = nn.Linear(4096, 487)
        
        self.gap = nn.AdaptiveAvgPool3d((1, 4, 4))
        self.fc = nn.Linear(8192, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        # Expects x to be (n, ch, fr, h, w).
        # Our x is (n, fr, ch, h, w)
        x = x.permute(0, 2, 1, 3, 4)
        assert (x.size()[2] >= 16)
        h = self.relu(self.conv1(x))
        h = self.pool1(h)
        h = self.relu(self.conv2(h))
        h = self.pool2(h)
        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)
        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)
        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        h = self.pool5(h)
        h = self.gap(h) # After the GAP, the dimensions are [1, 512, 1, 4, 4]
        h = h.view(-1, 8192)
        #h = self.relu(self.fc6(h))
        #h = self.dropout(h)
        #h = self.relu(self.fc7(h))
        #h = self.dropout(h)

        #logits = self.fc8(h)
        logits = self.fc(h)

        return logits 


model_dict = {'SeqAvgDenseNet121': SeqAvgDenseNet121,
              'SeqAvgResNet18': SeqAvgResNet18,
              'SeqAvgAlexNet': SeqAvgAlexNet,
              'SeqAvgVGG13': SeqAvgVGG13,
              'SeqAvgVGG16': SeqAvgVGG16,
              'DoublePoolAlexNet':DoublePoolAlexNet,
              'C3D': C3D} 
'''
              'R2Plus1DModel': R2Plus1DModel,
              'ThreeDResNet10': ThreeDResNet,
              'ThreeDResNet18': ThreeDResNet,
              'ThreeDResNet34': ThreeDResNet,
              'ThreeDResNet50': ThreeDResNet,
              'ThreeDResNet101': ThreeDResNet,
              'ThreeDResNet152': ThreeDResNet,
              'ThreeDResNet200': ThreeDResNet,
              'ThreeDResNeXt101': ThreeDResNeXt,
              'ThreeDWideResNet50': ThreeDWideResNet50,
              'ThreeDDenseNet121': ThreeDDenseNet121}
'''

