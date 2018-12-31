import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import math
import torch.autograd as autograd
import psutil
from r2plus1d import R2Plus1DModel

# model_dict = { 'resnet152': ResNet152, 'densenet121': DenseNet121,
#               'simple3d': Simple3D,
#               'SeqMaxResNet18': SeqMaxResNet18,
#               'SeqRNNResNet18': SeqRNNResNet18,
#               'Chest3DConv': Chest3DConv,
#               'resnet183D': Resnet183D,
#               'resnet343D': Resnet343D,
#               'resnet503D': Resnet503D,
#               'SeqRNNResNet34':SeqRNNResNet34,
#               'SeqRNNDenseNet121': SeqRNNDenseNet121,
#               'SeqRNNResNet18Hidden': SeqRNNResNet18Hidden,
#               'GenthialEncoder': GenthialEncoder,
#               'SeqAvgResNet18': SeqAvgResNet18,
#               'SeqAvgAlexNet': SeqAvgAlexNet,
#               'SeqAvgVGG13': SeqAvgVGG13,
#               'SeqAvgVGG16': SeqAvgVGG16}

class BDLSTM_resnet503D(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = Resnet503D(config, num_classes)
        self.lstm = nn.LSTM(input_size = 2048, hidden_size=512, num_layers=2, batch_first=False, dropout=0.2, bidirectional=True)
        self.config = config
        self.fc = nn.Linear(2048, num_classes)

    def get_features(self, x):
        x = self.model.conv1(x)

        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x):
        features_list = []
        CHUNK_SIZE = 4
        STEP_SIZE = 2
        start = 0
        while start + CHUNK_SIZE <= x.size()[2]:
            features_list.append(self.get_features(x[:, :, start:start + CHUNK_SIZE, :, :]))
            start += STEP_SIZE
        features_var = torch.stack(features_list, dim=0)
        rnn_out, (hidden, _) = self.lstm(features_var)
        hidden = hidden.permute(1, 0, 2)
        hidden = hidden.view(x.size()[0], -1)
        out = self.fc(hidden)
        return out


class Resnet503D(nn.Module):
    def __init__(self, config, num_classes):
        block = Bottleneck
        layers = [3,4,6,3]
        self.inplanes = 64
        super(Resnet503D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3,7,7), stride=(1,2,2), padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes * block.expansion),
                    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout3d(p=0.2)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Resnet343D(nn.Module):
    def __init__(self, config, num_classes):
        block = BasicBlock
        layers = [3,4,6,3]
        self.inplanes = 64
        super(Resnet343D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes),
                    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Resnet183D(nn.Module):
    def __init__(self, config, num_classes):
        block = BasicBlock
        layers = [2,2,2,2]
        self.inplanes = 64
        super(Resnet183D, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                    nn.Conv3d(self.inplanes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes),
                    )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

class Simple3D(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = nn.Sequential(
                                    nn.Conv3d(1, 4, 3),
                                    nn.Conv3d(4, 1, 3),
                                    nn.AdaptiveAvgPool3d(num_classes)
                                  )
        self.config = config

    def forward(self, X):
        y_hat = self.model(X)
        y_hat = y_hat.view((y_hat.size(0), 1))
        return y_hat


class Chest3DConv(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config

        self.input_layer = nn.Conv3d(3, 32, 5, stride=2, padding=2)
        self.conv1 = nn.Conv3d(32, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 192, 3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(192, 384, 3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(384, 256, 3, stride=1, padding=1)
        self.conv5 = nn.Conv3d(256, 256, 3, stride=1, padding=1)


        self.pool = nn.MaxPool3d(2, stride=2)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc1 = nn.Linear(256 , 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.avgpool = nn.AdaptiveAvgPool3d(1)

    def forward(self, X):
        X = self.input_layer(X)
        X = self.leaky_relu(X)
        X = self.conv1(X)
        X = self.leaky_relu(X)
        X = self.pool(X)
        X = self.conv2(X)
        X = self.leaky_relu(X)
        X = self.conv3(X)
        X = self.leaky_relu(X)
        X = self.pool(X)
        X = self.conv4(X)
        X = self.leaky_relu(X)
        X = self.conv5(X)
        X = self.leaky_relu(X)
        X = self.avgpool(X)
        X = X.view((X.size(0), -1))
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        return X

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
            residual = x

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                residual = self.downsample(x)

            out += residual
            out = self.relu(out)
            return out

class GenthialEncoder(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = nn.Sequential(nn.Conv2d(3, 64, (3,3)),
                nn.ReLU(),
                nn.MaxPool2d((3,3), stride=(2,2)),
                nn.Conv2d(64, 128, (3,3)),
                nn.ReLU(),
                nn.MaxPool2d((3,3), stride=(2,2)),
                nn.Conv2d(128, 256, (3,3)),
                nn.ReLU(),
                nn.Conv2d(256, 256, (3,3)),
                nn.ReLU(),
                nn.MaxPool2d((3,3), stride=(2,2)),
                nn.Conv2d(256, 512, (3,3)),
                nn.ReLU(),
                nn.MaxPool2d((3,3), stride=(2,2)))

        self.adapool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = 512

        #self.hidden_dim = 128

        #self.fc = nn.Linear(self.hidden_dim*1, num_classes)
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.config = config
        #self.dropout = nn.Dropout2d(p=0.2)
        #self.RNN = torch.nn.LSTM(input_size=num_ftrs,
                                 #hidden_size=self.hidden_dim,
                                 #num_layers=1,
                                 #batch_first=True,
                                 #bidirectional=False,
                                 #dropout=0.2)

    def get_features(self, X):
        x = self.model(X)
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
        features = features.permute((0, 2, 1))
        features_pooled = torch.nn.functional.adaptive_avg_pool1d(features, 1).view(batch_size, -1)
        out = self.fc(features_pooled)
        return out

class DenseNet121(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.densenet121(pretrained=config.pretrained)
        self.gap = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)
        self.config = config

    def forward(self, X):
        features = self.model.features(X)
        relu_out = nn.functional.relu(features, inplace=True)
        gap_out = self.gap(relu_out).view(features.size(0), -1)
        y_hat = self.model.classifier(gap_out)
        return y_hat

class SeqRNNDenseNet121(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.densenet121(pretrained=config.pretrained)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.classifier.in_features

        self.hidden_dim = 32

        self.fc = nn.Linear(self.hidden_dim*1, num_classes)
        self.config = config
        self.dropout = nn.Dropout2d(p=0.3)

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
        # self.dropout = nn.Dropout2d(p=0.0)
        self.RNN = torch.nn.LSTM(input_size=num_ftrs,
                                 hidden_size=self.hidden_dim,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=False,
                                 dropout=0.0)

    def get_features(self, X):
        x = self.model.features(X)
        x = nn.functional.relu(x, inplace=True)
        x = self.adapool(x)
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

class ResNet152(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.resnet152(pretrained=config.pretrained)
        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.config = config

    def forward(self, X):
        return self.model(X)

class SeqRNNResNet18Hidden(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.resnet18(pretrained=config.pretrained)
        self.adapool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features

        self.hidden_dim = 32

        self.fc = nn.Linear(self.hidden_dim*1, num_classes)
        self.config = config
        #self.dropout = nn.Dropout2d(p=0.2)
        self.RNN = torch.nn.LSTM(input_size=num_ftrs,
                                 hidden_size=self.hidden_dim,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=False,
                                 dropout=0.0)

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
        out = self.fc(self.hidden[1]).view(-1, 1)
        return out