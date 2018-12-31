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


class AlexNet(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.model = models.alexnet(pretrained=True)

        # num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
                                    nn.Dropout(),
                                    nn.Linear(256 * 6 * 6, 4096),
                                    nn.ReLU(inplace=True),
                                    nn.Dropout(),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(inplace=True),
                                    nn.AdaptiveAvgPool2d(1),
                                    nn.Linear(4096, num_classes))
    def forward(self, X):
        return self.model(X)

# Thanks to the ultrasound team for encoder models
class GenthialEncoder(nn.Module):
    #Inspired by Guillaume Genthial's encoder: https://github.com/guillaumegenthial/im2latex/blob/master/model/encoder.py


    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config
        self.do = config.dropout
        self.model = nn.Sequential(
                                    nn.Conv3d(1, 64, (1, 3, 3)),
                                    nn.ReLU(),
                                    nn.MaxPool3d((1,3,3), stride=(1, 2, 2)),
                                    nn.Dropout3d(p=self.do),
                                    nn.Conv3d(64, 128, (1, 3, 3)),
                                    nn.ReLU(),
                                    nn.MaxPool3d((1,3,3), stride=(1, 2, 2)),
                                    nn.Dropout3d(p=self.do),
                                    nn.Conv3d(128, 256, (1, 3, 3)),
                                    nn.ReLU(),
                                    nn.Conv3d(256, 256, (1, 3, 3)),
                                    nn.ReLU(),
                                    nn.MaxPool3d((1,3,3), stride=(1, 2, 2)),
                                    nn.Dropout3d(p=self.do),
                                    nn.Conv3d(256, 512, (1, 3, 3)),
                                    nn.ReLU(),
                                    nn.MaxPool3d((1,3,3), stride=(1, 2, 2)),
                                    nn.Dropout3d(p=self.do)
                                  )

        

    def forward(self, X):
        encoding = self.model(X)

        return encoding

def get_out_shape(input_shape, model):
    example_tensor = torch.Tensor(np.zeros(input_shape))
    example_tensor = example_tensor.unsqueeze(0)
    example_tensor = autograd.Variable(example_tensor)

    example_encoder_output = model(example_tensor)
    example_shape = list(example_encoder_output.size())
    return example_shape

class ManyToOneAvg(nn.Module):

    def __init__(self, config, num_classes):
        super().__init__()

        self.encoder = GenthialEncoder(config, num_classes)
        tensor_shape = np.zeros((1, 17, config.scale, config.scale)).shape

        # Create a fake tensor and run through encoder to get output shape of encoder
        batch_size, c, num_slices, h_new, w_new = get_out_shape(tensor_shape, self.encoder)
        self.last_size = c * h_new * w_new
        print("Last size: ", self.last_size)

        self.decoder = nn.Linear((self.last_size), num_classes)

    def forward(self, X):

        encoding = self.encoder(X)

        avg = torch.mean(encoding,2)
        avg = avg.view((-1, self.last_size))
        output = self.decoder(avg)

        return output

class ManyToOneMax(nn.Module):

    def __init__(self, config, num_classes):
        super().__init__()

        self.encoder = GenthialEncoder(config, num_classes)
        tensor_shape = np.zeros((1, 17, config.scale, config.scale)).shape

        # Create a fake tensor and run through encoder to get output shape of encoder
        batch_size, c, num_slices, h_new, w_new = get_out_shape(tensor_shape, self.encoder)
        self.last_size = c * h_new * w_new
        print("Last size: ", self.last_size)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.decoder = nn.Linear((self.last_size), num_classes)

    def forward(self, X):

        encoding = self.encoder(X)

        mxm = torch.max(encoding,2)[0]
        mxm = mxm.view((-1, self.last_size))
        output = self.decoder(mxm)

        return output

class ManyToOneLSTM(nn.Module):

    def __init__(self, config, num_classes):
        super().__init__()
        self.config = config

        self.encoder = GenthialEncoder(config, num_classes)
        tensor_shape = np.zeros((1, 17, config.scale, config.scale)).shape

        # Create a fake tensor and run through encoder to get output shape of encoder
        batch_size, c, num_slices, h_new, w_new = get_out_shape(tensor_shape, self.encoder)
        self.last_size = c * h_new * w_new
        print("Last size: ", self.last_size)
  
        self.hidden_dim = config.hidden_dim
        self.lstm = nn.LSTM(self.last_size, self.hidden_dim, batch_first = True, dropout=config.dropout)

        self.classifier = nn.Linear(self.hidden_dim, num_classes)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        hid = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        hid2 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        return (hid.cuda(), hid2.cuda())

    def forward(self, X):
        batch_size, depth, num_slices, rows, cols = X.size()
        encoding = self.encoder(X)
        # TODO : should we avgpool or not?
        encoder_out = encoding.view(batch_size, num_slices, -1)

        lstm_out, self.hidden = self.lstm(encoder_out, self.hidden)
        # lstm_out = lstm_out.view(1, 1, -1)
        # print("lstm_out: ", lstm_out.size())
        lstm_out = lstm_out[:,-1,:]
        # print("input to classifier: ", lstm_out.size())
        output = self.classifier(lstm_out)

        return output


class ResNet152(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.resnet152(pretrained=config.pretrained)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        self.config = config

    def forward(self, X):
        return self.model(X)

class DenseNet121(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()

        self.model = models.densenet121(pretrained=config.pretrained)
        self.dropout = nn.Dropout(0.2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, num_classes)
        self.config = config
        self.seq = config.seq

    def forward(self, X):
        # X is shape [batch_size, 3, num_slices, rows, cols]
        batch_size, depth, num_slices, rows, cols = X.size()
        # print('X shape: {}'.format(X.size()))
        # Reshape to [batch_size * num_slices, 3, rows, cols]
        X = X.view(-1, 3, rows, cols)
        # print('[batch_size * num_slices, 3, rows, cols]: {}'.format(X.size()))
        # pass reshaped X through NN
        features = self.model.features(X)
        relu_out = nn.functional.relu(features, inplace=True)
        # drop_out = self.dropout(relu_out)
        gap_out = self.gap(relu_out).view(features.size(0), -1)
        # print("gap out shape: ", gap_out.size())
        # print("feature shape: ", features.size())
        y_hat = self.model.classifier(gap_out)
        # print('y_hat: {}'.format(y_hat.size()))
        # reshape to [batch_size, num_slices, pred]
        y_hat = y_hat.view(batch_size, num_slices, 1)
        # print('[batch_size, num_slices, pred]: {}'.format(y_hat.size()))
        if self.seq == 'mean':
            pred = torch.mean(y_hat, dim=1)
        elif self.seq == 'max':
            pred = torch.max(y_hat, dim=1)[0]
        # print('predictions: {}'.format(pred.size()))
        return pred

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

class Deeper3D(nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        #Needs to be > 2
        self.num_conv_layers = 5
        self.dropout = True
        self.batchnorm = True
        self.max_pool = False
        num_conv_layers = self.num_conv_layers
        num_channels = 30
        kernel_size = 3
        pool_size = 6
        dropout_prob = .5
        for i in range(1, self.num_conv_layers + 1):
            strname = 'conv' + str(i)
            if i == 1: 
                setattr(self, strname, nn.Conv3d(1, num_channels, kernel_size = kernel_size))
            elif i == num_conv_layers:
                setattr(self, strname, nn.Conv3d(num_channels, 1, kernel_size = kernel_size))
            else:
                setattr(self, strname, nn.Conv3d(num_channels, num_channels, kernel_size = kernel_size))
            drpname = 'dropout' + strname
            batchname = 'batchnm' + strname
            setattr(self, drpname, nn.Dropout3d(p = dropout_prob))
            setattr(self, batchname, nn.BatchNorm3d(num_channels))
        self.aap = nn.AdaptiveAvgPool3d(pool_size)
        self.padding = nn.ReplicationPad3d((kernel_size//2))
        self.fc1 = nn.Linear(pool_size ** 3, num_classes)

    def forward(self, X):
        #print("FORWARD: ", torch.sum(X).data)
        for i in range(1, self.num_conv_layers + 1):
            name = 'conv' + str(i)
            drpname = 'dropout' + name
            batchname = 'batchnm' + name
            X = getattr(self, name)(X)
            if i == 2 and self.max_pool:
                X = F.max_pool3d(X, (1, 2, 2))
            if self.batchnorm and i != self.num_conv_layers:
                X = getattr(self, batchname)(X)
            X = F.relu(X)
            if self.dropout:
                X = getattr(self, drpname)((X))
            X = self.padding(X)
            #print("CONV LAYER {}. SHAPE: {}. SUM: {}".format(i, X.shape, torch.sum(X).data))
        X = self.aap(X)
        X = X.view(-1, self.num_flat_features(X))
        X = self.fc1(X)
        #print("OUTPUT SHAPE: {}, OUTPUT: {}".format(X.shape, X))
        return X

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

        #--fix_num_slices --num_slices 17 --model deeper3d
        
class LRCN(nn.Module):
    def __init__(self, config, num_classes, hidden_dim, dropout_p):
        super().__init__()
        self.config = config

        self.cnn_name = config.cnn[:3]
        self.cnn = cnn_dict[config.cnn]
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if self.cnn_name == 'res':            
            num_ftrs = self.cnn.fc.in_features
        elif self.cnn_name == 'den':
            num_ftrs = self.cnn.classifier.in_features
        else:
            raise RuntimeError('Double-check cnn argument')

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(num_ftrs, hidden_dim, batch_first = True, dropout=dropout_p)

        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(self.dropout_p)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.hidden = self.init_hidden()

    def densenet_forward(self, X, batch_size, num_slices):
        features = self.cnn.features(X)
        relu_out = F.relu(features, inplace=True)
        # drop_out = self.dropout(relu_out)
        cnn_out = self.avgpool(relu_out).view(batch_size, num_slices, -1)
        return cnn_out

    def resnet_forward(self, X, batch_size, num_slices):
        X = self.cnn.conv1(X)
        X = self.cnn.bn1(X)
        X = self.cnn.relu(X)
        X = self.cnn.maxpool(X)

        X = self.cnn.layer1(X)
        X = self.cnn.layer2(X)
        X = self.cnn.layer3(X)
        X = self.cnn.layer4(X)

        cnn_out = self.avgpool(X).view(batch_size, num_slices, -1)
        cnn_out2 = torch.unsqueeze(self.avgpool(X).view(batch_size, -1), dim=0)

        return cnn_out


    def init_hidden(self):
        hid = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        hid2 = autograd.Variable(torch.zeros(1, 1, self.hidden_dim))
        return (hid.cuda(), hid2.cuda())

    def forward(self, X):
        batch_size, depth, num_slices, rows, cols = X.size()
        # A = X[0, 0, 0, :, :]
        #print('BEFORE: ', X[0, 0, 0, :, :])
        if batch_size != 1:
            raise ValueError('Only implemented for batch size 1')
        X = X.permute(0, 2, 1, 3, 4)
        X = torch.squeeze(X, dim=0)
        #print(X.shape)
        # Verifying reshaping
        '''
        example_image = X.cpu().data
        example_image = example_image.type('torch.ByteTensor')
        example_image = example_image[15]
        print(type(example_image))
        utils.save_image(example_image, '/deep/group/aihc-bootcamp-winter2018/mbereket/medical-imaging-starter-pack/visualizations/example1.jpg')
        '''
        #print('AFTER: ', X[0, 0, :, :])
        # B = X[0, 0, :, :]
        # print(torch.sum(A.data - B.data))
        if self.cnn_name == 'res':            
            cnn_out = self.resnet_forward(X, batch_size, num_slices)
        elif self.cnn_name == 'den':
            cnn_out = self.densenet_forward(X, batch_size, num_slices)
        else:
            raise RuntimeError('Double-check cnn argument')

        # print("gap_out: ", gap_out.size())
        lstm_out, self.hidden = self.lstm(cnn_out, self.hidden)
        # lstm_out = lstm_out.view(1, 1, -1)
        # print("lstm_out: ", lstm_out.size())
        lstm_out = lstm_out[:,-1,:]
        # print("input to classifier: ", lstm_out.size())
        out = self.classifier(lstm_out)
        # print("out shape: ", out.size())
        # print("out:", out)

        return out

# Make sure to add to this as you write models.
model_dict = {'resnet152': ResNet152,
              'densenet121': DenseNet121,
              'deeper3d' : Deeper3D,
              'simple3d': Simple3D,
              'lrcn': LRCN,
              'mtoavg': ManyToOneAvg,
              'mtomax': ManyToOneMax,
              'mtolstm': ManyToOneLSTM}

cnn_dict = {'resnet152': models.resnet152(pretrained=True),
            'resnet18': models.resnet18(pretrained=True),
            'resnet34': models.resnet34(pretrained=True),
            'resnet50': models.resnet50(pretrained=True),
            'resnet101': models.resnet101(pretrained=True),
            'densenet121': models.densenet121(pretrained=True),
            'densenet169': models.densenet169(pretrained=True),
            'densenet161': models.densenet161(pretrained=True),
            'densenet201': models.densenet201(pretrained=True)}