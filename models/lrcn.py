import re
import torch
import torch.nn as nn
import torch.nn.utils.rnn as utils
import torch.nn.functional as F

from torchvision import models


class LRCN(nn.Module):
    """LCRN Model.
    LRCN consists of a CNN whose outputs are fed into a stack of LSTMs. Both the CNN and
    LSTM weights are shared across time, so the model scales to arbitrarily long inputs.

    Based on the paper:
    "Long-term Recurrent Convolutional Networks for Visual Recognition and Description"
    by Jeff Donahue, Lisa Anne Hendricks, Marcus Rohrbach, Subhashini Venugopalan, Sergio Guadarrama, Kate Saenko, Trevor Darrell
    (https://arxiv.org/abs/1411.4389).
    """
    def __init__(self, hidden_dim=128, num_slices=8, dropout_prob=0.1, cnn_name='resnet152', num_classes=1, **kwargs):
        super(LRCN, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_slices = num_slices
        self.dropout_prob = dropout_prob
        self.cnn_name = cnn_name
        self.num_classes = num_classes
        self.cnn = cnn_dict[self.cnn_name](pretrained=True)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if self.cnn_name.startswith('resnet'):
            self.num_ftrs = self.cnn.fc.in_features
        elif self.cnn_name.startswith('densenet'):
            self.num_ftrs = self.cnn.classifier.in_features
        else:
            raise RuntimeError('Error: CNN argument invalid')

        self.lstm = nn.LSTM(self.num_ftrs, self.hidden_dim, batch_first=True, dropout=self.dropout_prob)
        self.classifier = nn.Linear(self.hidden_dim, 1)

    # TODO: Fix this and run
    def _densenet_forward(self, x):
        features = self.cnn.features(x)
        relu_out = F.relu(features, inplace=True)
        cnn_out = self.avgpool(relu_out).view(-1, self.num_slices, self.num_ftrs)
        return cnn_out

    def _resnet_forward(self, x):
        x = self.cnn.conv1(x)
        x = self.cnn.bn1(x)
        x = self.cnn.relu(x)
        x = self.cnn.maxpool(x)

        x = self.cnn.layer1(x)
        x = self.cnn.layer2(x)
        x = self.cnn.layer3(x)
        x = self.cnn.layer4(x)

        return self.gap(x).view(-1, self.num_slices, self.num_ftrs)

    def forward(self, inputs, device):
        """Forward Pass. Expects inputs of shape Batch_size x Channels x Slices x Height x Width.
        Inputs should be sorted by sequence length along the batch dimension, and the lengths vector
        should contain the lengths of each sequence."""
        
        # Reshape input to allow for simultaneous processing of all slices.
        data = inputs.batch
        lengths = inputs.lengths

        B, C, S, H, W = data.shape
        inputs_t = torch.transpose(data, 1, 2).contiguous()
        cnn_inputs = inputs_t.view(B * S, C, H, W)

        # Make CNN forward pass.
        if self.cnn_name.startswith('resnet'):
            features = self._resnet_forward(cnn_inputs)
        elif self.cnn_name.startswith('densenet'):
            features = self._densenet_forward(cnn_inputs)
        else:
            raise RuntimeError('Double-check cnn argument')

        # Reshape back into sequences and make LSTM forward pass.
        lstm_inputs = features.reshape(data.shape[0], data.shape[2], self.num_features)
        lstm_packed = utils.pack_padded_sequence(lstm_inputs, lengths, batch_first=True)
        lstm_out, _ = self.lstm(lstm_packed)
        lstm_unpacked, _ = utils.pad_packed_sequence(lstm_out, batch_first=True)

        # Extract final outputs for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(lstm_unpacked.shape[0], lstm_unpacked.shape[2]).unsqueeze(1)
        lstm_final_outputs = lstm_unpacked.gather(1, idx).squeeze()

        logits = self.classifier(lstm_final_outputs)
        return logits

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `LRCN(**model_args)`.
        """
        model_args = {'hidden_dim': self.hidden_dim,
                      'num_slices': self.num_slices,
                      'dropout_prob': self.dropout_prob,
                      'cnn_name': self.cnn_name,
                      'num_classes': self.num_classes,
                      'num_layers': self.num_layers}

        return model_args


cnn_dict = {'resnet152': models.resnet152,
            'resnet18': models.resnet18,
            'resnet34': models.resnet34,
            'resnet50': models.resnet50,
            'resnet101': models.resnet101,
            'densenet121': models.densenet121,
            'densenet169': models.densenet169,
            'densenet161': models.densenet161,
            'densenet201': models.densenet201}
