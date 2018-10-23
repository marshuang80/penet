"""
model.py
    Defines PyTorch nn.Module classes.
    Each should implement a constructor __init__(self, config)
    and a forward pass forward(self, x)
"""
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import math
import torch.autograd as autograd
import psutil


class SeqMaxResNet18(nn.Module):
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
        features_pooled = torch.nn.functional.adaptive_max_pool1d(features, 1).view(batch_size, -1)
        out = self.fc(features_pooled)
        return out

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
        # TODO: Should we have relu here?
        # out = F.relu(features, inplace=True)
        
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


class SeqRNNResNet18(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.resnet18(pretrained=config.pretrained)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features

        self.hidden_dim = 128

        self.fc = nn.Linear(self.hidden_dim*1, num_classes)
        self.config = config
        #self.dropout = nn.Dropout2d(p=0.2)
        self.RNN = torch.nn.LSTM(input_size=num_ftrs,
                                 hidden_size=self.hidden_dim,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=False,
                                 dropout=0.2)

    def get_features(self, X):
        x = self.model.conv1(X)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        #x = self.dropout(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.adapool(x)
        x = x.view(x.size(0), -1)
        return x

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        self.hidden = (autograd.Variable(torch.zeros(1, 1, self.hidden_dim).cuda()),
                       autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())

    def forward(self, X):
        # Expect input of size [Batch_size, #Images, #Channels, H, W]

        batch_size, n_images, n_channels, h, w = X.size()
        X = X.view(-1, 3, h, w)
        features = self.get_features(X)
        features = features.view(batch_size, n_images, -1)
        self.init_hidden()
        rnn_out, self.hidden = self.RNN(features, self.hidden)
        rnn_out = rnn_out[:, -1, :]
        out = self.fc(rnn_out)
        return out


class SeqRNNResNet34(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.resnet34(pretrained=config.pretrained)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.fc = nn.Linear(128*2, num_classes)
        self.config = config
        self.dropout = nn.Dropout2d(p=0.3)
        self.RNN = torch.nn.LSTM(input_size=num_ftrs,
                                 hidden_size=128,
                                 num_layers=1,
                                 batch_first=False,
                                 bidirectional=True,
                                 dropout=0.2)

    def get_features(self, X):
        x = self.model.conv1(X)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.dropout(x)
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
        features_list = []
        for i in range(X.size()[1]):
            features_list.append(self.get_features(X[:, i, :, :, :]))

        features_var = torch.stack(features_list, dim=0)

        rnn_out, (hidden, _) = self.RNN(features_var)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.view(X.size()[0], -1)

        out = self.fc(hidden)
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

        # (0): Conv2d (3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (1): ReLU(inplace)
        # (2): Conv2d (64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (3): ReLU(inplace)
        # (4): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        # (5): Conv2d (64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (6): ReLU(inplace)
        # (7): Conv2d (128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (8): ReLU(inplace)
        # (9): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        # (10): Conv2d (128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (11): ReLU(inplace)
        # (12): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (13): ReLU(inplace)
        # (14): Conv2d (256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (15): ReLU(inplace)
        # (16): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        # (17): Conv2d (256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (18): ReLU(inplace)
        # (19): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (20): ReLU(inplace)
        # (21): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (22): ReLU(inplace)
        # (23): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))
        # (24): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (25): ReLU(inplace)
        # (26): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (27): ReLU(inplace)
        # (28): Conv2d (512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # (29): ReLU(inplace)
        # (30): MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1))

        # import pdb; pdb.set_trace()
        # x = self.model.features(X)
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
        # import pdb; pdb.set_trace()
        # features = features.permute((0,2,1))
        # features_pooled = torch.nn.functional.adaptive_avg_pool1d(features, 1).view(batch_size, -1)
        out = self.fc(features_pooled)
        return out


model_dict = {'SeqMaxResNet18': SeqMaxResNet18,
              'SeqAvgResNet18': SeqAvgResNet18,
              'SeqAvgAlexNet': SeqAvgAlexNet,
              'SeqAvgVGG13': SeqAvgVGG13,
              'SeqRNNResNet18': SeqRNNResNet18,
              'SeqRNNResNet34': SeqRNNResNet34,
              'SeqAvgVGG16': SeqAvgVGG16,
  	      'DoublePoolAlexNet':DoublePoolAlexNet}
