import math
import torch
import torch.nn as nn
import util

from models.layers.penet import *


class PENetClassifier(nn.Module):
    """PENet stripped down for classification.

    The idea is to pre-train this network, then use the pre-trained
    weights for the encoder in a full PENet.
    """

    def __init__(self, model_depth, cardinality=32, num_channels=3, num_classes=600, init_method=None, **kwargs):
        super(PENetClassifier, self).__init__()

        self.in_channels = 64
        self.model_depth = model_depth
        self.cardinality = cardinality
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.in_conv = nn.Sequential(nn.Conv3d(self.num_channels, self.in_channels, kernel_size=7,
                                               stride=(1, 2, 2), padding=(3, 3, 3), bias=False),
                                     nn.GroupNorm(self.in_channels // 16, self.in_channels),
                                     nn.LeakyReLU(inplace=True))
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        # Encoders
        if model_depth != 50:
            raise ValueError('Unsupported model depth: {}'.format(model_depth))
        encoder_config = [3, 4, 6, 3]
        total_blocks = sum(encoder_config)
        block_idx = 0

        self.encoders = nn.ModuleList()
        for i, num_blocks in enumerate(encoder_config):
            out_channels = 2 ** i * 128
            stride = 1 if i == 0 else 2
            encoder = PENetEncoder(self.in_channels, out_channels, num_blocks, self.cardinality,
                                  block_idx, total_blocks, stride=stride)
            self.encoders.append(encoder)
            self.in_channels = out_channels * PENetBottleneck.expansion
            block_idx += num_blocks

        self.classifier = GAPLinear(self.in_channels, num_classes)

        if init_method is not None:
            self._initialize_weights(init_method, focal_pi=0.01)

    def _initialize_weights(self, init_method, gain=0.2, focal_pi=None):
        """Initialize all weights in the network."""
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear):
                if init_method == 'normal':
                    nn.init.normal_(m.weight, mean=0, std=gain)
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=gain)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight)
                else:
                    raise NotImplementedError('Invalid initialization method: {}'.format(self.init_method))
                if hasattr(m, 'bias') and m.bias is not None:
                    if focal_pi is not None and hasattr(m, 'is_output_head') and m.is_output_head:
                        # Focal loss prior (~0.01 prob for positive, see RetinaNet Section 4.1)
                        nn.init.constant_(m.bias, -math.log((1 - focal_pi) / focal_pi))
                    else:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm) and m.affine:
                # Gamma for last GroupNorm in each residual block gets set to 0
                init_gamma = 0 if hasattr(m, 'is_last_norm') and m.is_last_norm else 1
                nn.init.constant_(m.weight, init_gamma)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # Expand input (allows pre-training on RGB videos, fine-tuning on Hounsfield Units)
        if x.size(1) < self.num_channels:
            x = x.expand(-1, self.num_channels // x.size(1), -1, -1, -1)

        x = self.in_conv(x)
        x = self.max_pool(x)

        # Encoders
        for encoder in self.encoders:
            x = encoder(x)

        # Classifier
        x = self.classifier(x)

        return x

    def args_dict(self):
        """Get a dictionary of args that can be used to reconstruct this architecture.
        To use the returned dict, initialize the model with `PENet(**model_args)`.
        """
        model_args = {
            'model_depth': self.model_depth,
            'cardinality': self.cardinality,
            'num_classes': self.num_classes,
            'num_channels': self.num_channels
        }

        return model_args

    def load_pretrained(self, ckpt_path, gpu_ids):
        """Load parameters from a pre-trained PENetClassifier from checkpoint at ckpt_path.
        Args:
            ckpt_path: Path to checkpoint for PENetClassifier.
        Adapted from:
            https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
        """
        device = 'cuda:{}'.format(gpu_ids[0]) if len(gpu_ids) > 0 else 'cpu'
        pretrained_dict = torch.load(ckpt_path, map_location=device)['model_state']
        model_dict = self.state_dict()

        # Filter out unnecessary keys
        pretrained_dict = {k[len('module.'):]: v for k, v in pretrained_dict.items()}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)

        # Load the new state dict
        self.load_state_dict(model_dict)

    def fine_tuning_parameters(self, fine_tuning_boundary, fine_tuning_lr=0.0):
        """Get parameters for fine-tuning the model.
        Args:
            fine_tuning_boundary: Name of first layer after the fine-tuning layers.
            fine_tuning_lr: Learning rate to apply to fine-tuning layers (all layers before `boundary_layer`).
        Returns:
            List of dicts that can be passed to an optimizer.
        """

        def gen_params(boundary_layer_name, fine_tuning):
            """Generate parameters, if fine_tuning generate the params before boundary_layer_name.
            If unfrozen, generate the params at boundary_layer_name and beyond."""
            saw_boundary_layer = False
            for name, param in self.named_parameters():
                if name.startswith(boundary_layer_name):
                    saw_boundary_layer = True

                if saw_boundary_layer and fine_tuning:
                    return
                elif not saw_boundary_layer and not fine_tuning:
                    continue
                else:
                    yield param

        # Fine-tune the network's layers from encoder.2 onwards
        optimizer_parameters = [{'params': gen_params(fine_tuning_boundary, fine_tuning=True), 'lr': fine_tuning_lr},
                                {'params': gen_params(fine_tuning_boundary, fine_tuning=False)}]

        # Debugging info
        util.print_err('Number of fine-tuning layers: {}'
                       .format(sum(1 for _ in gen_params(fine_tuning_boundary, fine_tuning=True))))
        util.print_err('Number of regular layers: {}'
                       .format(sum(1 for _ in gen_params(fine_tuning_boundary, fine_tuning=False))))

        return optimizer_parameters
