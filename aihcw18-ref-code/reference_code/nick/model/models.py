"""
model.py
    Defines PyTorch nn.Module classes.
    Each should implement a constructor __init__(self, config)
    and a forward pass forward(self, x)
"""
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch
from torchvision import models
import torch.nn.functional as F
from torchvision import utils
import numpy as np

class Simple2D(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        if config.rgb == True:
            raise ValueError('rgb should be false')
        self.model = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2),
                        nn.Conv2d(128, 256, kernel_size=3),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2))
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, X):
        batch_size, num_slices, depth, rows, cols = X.size()
        if batch_size != 1:
            raise ValueError('Only implemented for batch size 1')
        X = torch.squeeze(X, dim=0)
        X = X.permute(1, 0, 2, 3)
        features = self.model(X)
        X = F.relu(features, inplace=True)
        X = self.maxpool(X).view(features.size(0), -1)
        out = torch.mean(X,0, keepdim=True)
        out = self.classifier(out)
        return out

class Simple2DMultiview(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        if config.rgb == True:
            raise ValueError('rgb should be false')

        sag_model = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2),
                        nn.Conv2d(128, 256, kernel_size=3),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2))

        cor_model = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2),
                        nn.Conv2d(128, 256, kernel_size=3),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2))

        ax_model = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size=3),
                        nn.BatchNorm2d(64),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2),
                        nn.Conv2d(64, 128, kernel_size=3),
                        nn.BatchNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2),
                        nn.Conv2d(128, 256, kernel_size=3),
                        nn.BatchNorm2d(256),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(3, stride=2))
 
        sag_model = sag_model.cuda()
        cor_model = cor_model.cuda()
        ax_model = ax_model.cuda()

        self.models = {
                'sagittal': sag_model,
                'coronal': cor_model,
                'axial': ax_model,
                }

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(256 * 3, num_classes)

    def forward(self, X_sag, X_cor, X_ax):
        X_dict = {
                'sagittal': X_sag,
                'coronal': X_cor,
                'axial': X_ax,
                }
        outs = []
        for view in ['sagittal', 'coronal', 'axial']:
            X = X_dict[view]
            batch_size, num_slices, depth, rows, cols = X.size()
            X = torch.squeeze(X, dim=0)
            #print(view + ' X', X.size())
            X = X.permute(1, 0, 2, 3)
            features = self.models[view](X)
            #print(view + 'features', features.size())
            if self.config.pooling in ['avgmax', 'avgavg']:
                out = self.avgpool(features).view(features.size(0), -1)
            else:
                out = self.maxpool(features).view(features.size(0), -1)
            #print(view + ' after avg pool', out.size())

            if self.config.pooling in ['avgmax', 'maxmax']:
                out = torch.max(out, 0, keepdim=True)[0]
            else:
                out = torch.mean(out, 0, keepdim=True)
            #print(view + ' after max', out.size())
            outs.append(out)

        concat_outs = torch.cat(outs, 1)
        #print('concat_outs', concat_outs.size())

        classifier_out = self.classifier(concat_outs)
        #print('classifier_out', classifier_out.size())
        return classifier_out


class CNN(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.pooling = config.pooling
        #self.add_relu = config.add_relu
        #assert self.pooling in ['avgavg','avgmax','maxavg','maxmax'], 'Invalid pooling argument'
        
        #if self.add_relu:
        #    self.relu = nn.ReLU()
        self.cnn_name = config.cnn[:3]

        if config.pretrained:
            self.cnn = cnn_dict_pretrained[config.cnn]
        else:
            self.cnn = cnn_dict_not_pretrained[config.cnn]
        
        self.cnn = cnn_dict_pretrained[config.cnn]
        
        for param in list(self.cnn.parameters())[:15]:

            param.requires_grad = False

        for name, param in self.cnn.named_parameters():
            if not param.requires_grad:
                print(name)
        
        # self.pool = nn.AdaptiveAvgPool2d(1) if self.pooling in ['avgavg','avgmax'] else nn.AdaptiveMaxPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        if self.cnn_name == 'res':            
            num_ftrs = self.cnn.fc.in_features
        elif self.cnn_name == 'den':
            num_ftrs = self.cnn.classifier.in_features
        else:
            raise RuntimeError('cnn currently only supports resnets and densenets')

        if config.return_layer == 'layer4':
            num_ftrs = 512
        elif config.return_layer == 'layer3':
            num_ftrs = 256
        elif config.return_layer == 'layer2':
            num_ftrs = 128

        self.classifier = nn.Linear(num_ftrs, num_classes)

    def densenet_forward(self, X):
        features = self.cnn.features(X)

        # TODO: should we have relu here or not?
        # cnn_out = F.relu(features, inplace=True)
        return features

    def resnet_forward(self, X):
        print('initial size', X.size())
        X = self.cnn.conv1(X)
        print('after conv1', X.size())
        X = self.cnn.bn1(X)
        print('after bn1', X.size())
        X = self.cnn.relu(X)
        X = self.cnn.maxpool(X)
        print('after maxpool', X.size())

        X = self.cnn.layer1(X)
        print('after layer 1', X.size())
        X = self.cnn.layer2(X)
        print('after layer 2', X.size())
        
        if self.config.return_layer == 'layer2':
            return X
        
        X = self.cnn.layer3(X)
        print('after layer 3', X.size())

        if self.config.return_layer == 'layer3':
            return X

        X = self.cnn.layer4(X)
        print('after layer 4', X.size())

        if self.config.return_layer == 'layer4':
            return X

        #X = self.avg_pool(X)
        #print('after avg pool', X.size())

        #X = X.view(X.size(0), -1)

        # TODO: should we have relu here or not?
        # cnn_out = F.relu(features, inplace=True)
        
        return X

    def forward(self, X):
        batch_size, num_slices, depth, rows, cols = X.size()
        if batch_size != 1:
            raise ValueError('Only implemented for batch size 1')
        X = torch.squeeze(X, dim=0)

        if self.cnn_name == 'res':            
            features = self.resnet_forward(X)
        elif self.cnn_name == 'den':
            features = self.densenet_forward(X)
        else:
            raise RuntimeError('Double-check cnn argument')

        #if self.add_relu:
        #    features = self.relu(features)

        print('features', features.size())
        out = self.avg_pool(features).view(features.size(0), -1)
        print('after avg pool', out.size())
        out = torch.max(out, 0, keepdim=True)[0]
        print('after max', out.size())
        out = self.classifier(out)
        return out


class AlexNet(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.pooling = config.pooling
        assert self.pooling in ['avgavg','avgmax','maxavg','maxmax'], 'Invalid pooling argument'
        self.model = models.alexnet(pretrained=config.pretrained)
        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        '''
        # num_features = list(self.model.classifier.children())[-1].in_features
        # new_classifier = nn.Sequential(*list(self.model.classifier.children())[:-1], nn.Linear(num_features, num_classes))
        # self.model.classifier = new_classifier
        self.pool = nn.AdaptiveAvgPool2d(1) if self.pooling in ['avgavg','avgmax'] else nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, X):
        batch_size, num_slices, depth, rows, cols = X.size()
        if batch_size != 1:
            raise ValueError('Only implemented for batch size 1')
        #print('original shape: ', X.size())
        X = torch.squeeze(X, dim=0)
        #print('squeeze batch_size: ', X.size())
        features = self.model.features(X)
        #print('after Alexnet features: ', features.size())
        
        # TODO: Should we have relu here?
        # out = F.relu(features, inplace=True)
        
        if self.pooling in ['avgavg','maxavg']:
            out = self.pool(features).view(features.size(0), -1)
            #print('after pool2d(1): ', out.size())
            out = torch.mean(out, 0, keepdim=True)
            #print('after avging axis 0: ', out.size())
        else:
            out = self.pool(features).view(features.size(0), -1)
            out = torch.max(out, 0, keepdim=True)[0]

        out = self.classifier(out)
        #print('after linear classifier: ', out.size())
        return out

class Multiview(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        sag_model = models.alexnet(pretrained=config.pretrained)
        sag_model = sag_model.cuda()
        cor_model = models.alexnet(pretrained=config.pretrained)
        cor_model = cor_model.cuda()
        ax_model = models.alexnet(pretrained=config.pretrained)
        ax_model = ax_model.cuda()

        self.models = {
                'sagittal': sag_model,
                'coronal': cor_model,
                'axial': ax_model,
                }
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        #self.lstm = torch.nn.LSTM(input_size=256, hidden_size=256, batch_first=True, dropout=0.2, bidirectional=True)
        self.classifier = nn.Linear(256 * 3, num_classes)
        #self.classifier = nn.Sequential(
        #    nn.Dropout(),
        #    nn.Linear(256 * 3, 256),
        #    nn.ReLU(inplace=True),
        #    nn.Dropout(),
        #    nn.Linear(256, num_classes),
        #)
        #self.classifier = nn.Sequential(
        #    nn.Dropout(),
        #    nn.Linear(256*8*8*3, 4096

    def forward(self, X_sag, X_cor, X_ax):
        X_dict = {
                'sagittal': X_sag,
                'coronal': X_cor,
                'axial': X_ax,
                }
        outs = []
        for view in ['sagittal', 'coronal', 'axial']:
            X = X_dict[view]
            batch_size, num_slices, depth, rows, cols = X.size()
            X = torch.squeeze(X, dim=0)
            #print(view + ' X', X.size())
            features = self.models[view].features(X)
            #print(view + 'features', features.size())
            if self.config.pooling in ['avgmax', 'avgavg']:
                out = self.avgpool(features).view(features.size(0), -1)
            else:
                out = self.maxpool(features).view(features.size(0), -1)
            #print(view + ' after avg pool', out.size())
            
            # LSTM
            #out = torch.unsqueeze(out, dim=0)
            #print(view + ' after unsqueeze', out.size())
            #out, _ = self.lstm(out)
            #print(view + ' after lstm', out.size())
            #out = torch.squeeze(out, dim=0)
            #print(view + ' after squeeze', out.size())
            if self.config.pooling in ['avgmax', 'maxmax']:
                out = torch.max(out, 0, keepdim=True)[0]
            else:
                out = torch.mean(out, 0, keepdim=True)
            #print(view + ' after max', out.size())
            outs.append(out)
        
        concat_outs = torch.cat(outs, 1)
        #print('concat_outs', concat_outs.size())
        
        classifier_out = self.classifier(concat_outs)
        #print('classifier_out', classifier_out.size())
        return classifier_out

class DenseNet121(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.model = models.densenet121(pretrained=config.pretrained)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.classifier.in_features
        self.classifier = nn.Linear(num_ftrs, num_classes)
        self.config = config
        self.seq = config.seq

    def forward(self, X):
        batch_size, num_slices, depth, rows, cols = X.size()
        if batch_size != 1:
            raise ValueError('Only implemented for batch size 1')
        print('original shape: ', X.size())
        X = torch.squeeze(X, dim=0)
        print('squeeze batch_size: ', X.size())
        features = self.model.features(X)
        print('after Densenet features: ', features.size())
        out = F.relu(features, inplace=True)
        print('after relu: ', out.size())
        out = self.avgpool(out).view(out.size(0), -1)
        print('after avgpool: ', out.size())
        out = torch.mean(out, 0, keepdim=True)
        print('after avging axis 0: ', out.size())
        out = self.classifier(out)
        print('after linear classifier: ', out.size())
        return out

class ResNet18(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.model = models.resnet18(pretrained=config.pretrained)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, X):
        batch_size, num_slices, depth, rows, cols = X.size()
        if batch_size != 1:
            raise ValueError('Only implemented for batch size 1')
        X = torch.squeeze(X, dim=0)
        X = self.model.conv1(X)
        X = self.model.bn1(X)
        X = self.model.relu(X)
        X = self.model.maxpool(X)

        X = self.model.layer1(X)
        X = self.model.layer2(X)
        X = self.model.layer3(X)
        X = self.model.layer4(X)

        out = self.maxpool(X).view(X.size(0), -1)

        avg = torch.mean(out,0, keepdim=True)
        # print('avg: ', avg.size())
        out = self.classifier(avg)
        return out

class VGG(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.model = models.vgg11_bn(pretrained=config.pretrained)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        batch_size, num_slices, depth, rows, cols = x.size()
        x = torch.squeeze(x, dim=0)

        x = self.model.features(x)

        print('before avg pool: ', x.size())
        out = self.avg_pool(x).view(x.size(0), -1)
        print('after avg pool: ', out.size())
        out = torch.mean(out, 0, keepdim=True)
        print('after mean: ', out.size())

        out = self.classifier(out)

        return out

class Inception3(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.model = models.inception_v3(True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        # 299 x 299 x 3
        x = self.model.Conv2d_1a_3x3(x)
        # 149 x 149 x 32
        x = self.model.Conv2d_2a_3x3(x)
        # 147 x 147 x 32
        x = self.model.Conv2d_2b_3x3(x)
        # 147 x 147 x 64
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 73 x 73 x 64
        x = self.model.Conv2d_3b_1x1(x)
        # 73 x 73 x 80
        x = self.model.Conv2d_4a_3x3(x)
        # 71 x 71 x 192
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # 35 x 35 x 192
        x = self.model.Mixed_5b(x)
        # 35 x 35 x 256
        x = self.model.Mixed_5c(x)
        # 35 x 35 x 288
        x = self.model.Mixed_5d(x)
        # 35 x 35 x 288
        x = self.model.Mixed_6a(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6b(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6c(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6d(x)
        # 17 x 17 x 768
        x = self.model.Mixed_6e(x)
        # 17 x 17 x 768
        x = self.model.Mixed_7a(x)
        # 8 x 8 x 1280
        x = self.model.Mixed_7b(x)
        # 8 x 8 x 2048
        x = self.model.Mixed_7c(x)

        x = self.avg_pool(x)

        x = torch.max(out = torch.max(out, 0, keepdim=True)[0])

        x = self.classifier(x)

        return x

# Make sure to add to this as you write models.
cnn_dict_pretrained = {'resnet152': models.resnet152(pretrained=True),
            'resnet18': models.resnet18(pretrained=True),
            'resnet34': models.resnet34(pretrained=True),
            'resnet50': models.resnet50(pretrained=True),
            'resnet101': models.resnet101(pretrained=True),
            'densenet121': models.densenet121(pretrained=True),
            'densenet169': models.densenet169(pretrained=True),
            'densenet161': models.densenet161(pretrained=True),
            'densenet201': models.densenet201(pretrained=True)}

cnn_dict_not_pretrained = {'resnet152': models.resnet152(pretrained=False),
            'resnet18': models.resnet18(pretrained=False),
            'resnet34': models.resnet34(pretrained=False),
            'resnet50': models.resnet50(pretrained=False),
            'resnet101': models.resnet101(pretrained=False),
            'densenet121': models.densenet121(pretrained=False),
            'densenet169': models.densenet169(pretrained=False),
            'densenet161': models.densenet161(pretrained=False),
            'densenet201': models.densenet201(pretrained=False)}

# Make sure to add to this as you write models.
model_dict = {'resnet18': ResNet18,
              'densenet121': DenseNet121,
              'alexnet' : AlexNet,
              'simple2d': Simple2D,
              'cnn': CNN,
              'vgg': VGG,
              'inception': Inception3,
              'multiview': Multiview,
              'simple2dmultiview': Simple2DMultiview,
              }
