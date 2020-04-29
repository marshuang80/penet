import util

from .base_arg_parser import BaseArgParser


class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True
        self.parser.add_argument('--epochs_per_save', type=int, default=5,
                                 help='Number of epochs between saving a checkpoint to save_dir.')
        self.parser.add_argument('--iters_per_print', type=int, default=4,
                                 help='Number of iterations between printing loss to the console and TensorBoard.')
        self.parser.add_argument('--epochs_per_eval', type=int, default=1,
                                 help='Number of epochs between evaluating model on the validation set.')
        self.parser.add_argument('--iters_per_visual', type=int, default=80,
                                 help='Number of iterations between visualizing training examples.')
        self.parser.add_argument('--learning_rate', type=float, default=1e-1,
                                 help='Initial learning rate.')
        self.parser.add_argument('--lr_scheduler', type=str, default='cosine_warmup',
                                 choices=('step', 'multi_step', 'plateau', 'cosine_warmup'),
                                 help='LR scheduler to use.')
        self.parser.add_argument('--lr_decay_gamma', type=float, default=0.1,
                                 help='Multiply learning rate by this value every LR step (step and multi_step only).')
        self.parser.add_argument('--lr_decay_step', type=int, default=300000,
                                 help='Number of epochs (iters) between each multiply-by-gamma step.')
        self.parser.add_argument('--lr_warmup_steps', type=int, default=10000,
                                 help='Number of iterations for linear LR warmup.')
        self.parser.add_argument('--lr_milestones', type=str, default='50,125,250',
                                 help='Epochs to step the LR when using multi_step LR scheduler.')
        self.parser.add_argument('--patience', type=int, default=10,
                                 help='Number of stagnant epochs before stepping LR.')
        self.parser.add_argument('--num_epochs', type=int, default=300,
                                 help='Number of epochs to train. If 0, train forever.')
        self.parser.add_argument('--max_ckpts', type=int, default=2,
                                 help='Number of recent ckpts to keep before overwriting old ones.')
        self.parser.add_argument('--best_ckpt_metric', type=str, default='val_loss', choices=('val_loss', 'val_AUROC'),
                                 help='Metric used to determine which checkpoint is best.')
        self.parser.add_argument('--max_eval', type=int, default=-1,
                                 help='Max number of examples to evaluate from the training set.')
        self.parser.add_argument('--optimizer', type=str, default='sgd', choices=('sgd', 'adam'), help='Optimizer.')
        self.parser.add_argument('--sgd_momentum', type=float, default=0.9, help='SGD momentum (SGD only).')
        self.parser.add_argument('--sgd_dampening', type=float, default=0.9, help='SGD momentum (SGD only).')
        self.parser.add_argument('--adam_beta_1', type=float, default=0.9, help='Adam beta 1 (Adam only).')
        self.parser.add_argument('--adam_beta_2', type=float, default=0.999, help='Adam beta 2 (Adam only).')
        self.parser.add_argument('--weight_decay', type=float, default=5e-4,
                                 help='Weight decay (i.e., L2 regularization factor).')
        self.parser.add_argument('--dropout_prob', type=float, default=0.0, help='Dropout probability.')
        self.parser.add_argument('--hidden_dim', type=float, default=32,
                                 help='LSTM hidden state size (LRCN only).')
        self.parser.add_argument('--elastic_transform', type=util.str_to_bool, default=False,
                                 help='Apply elastic transformation to input volume at training time.')
        self.parser.add_argument('--do_hflip', type=util.str_to_bool, default=True,
                                 help='If true, do random horizontal flip during training.')
        self.parser.add_argument('--do_vflip', type=util.str_to_bool, default=False,
                                 help='If true, do random vertical flip during training.')
        self.parser.add_argument('--do_rotate', type=util.str_to_bool, default=True,
                                 help='If true, do random rotation (up to +/- 15 degrees) of the scan during training.')
        self.parser.add_argument('--do_jitter', type=util.str_to_bool, default=True,
                                 help='If true, do random jitter of starting slices during training.')
        self.parser.add_argument('--do_center_pe', type=util.str_to_bool, default=True,
                                 help='If true, center training windows on the PE abnormality.')
        self.parser.add_argument('--abnormal_prob', type=float, default=0.5,
                                 help='Probability of sampling an abnormal window during training.')
        self.parser.add_argument('--use_pretrained', type=util.str_to_bool, default=False,
                                 help='If True, load a pretrained model from ckpt_path (PENet loading PENetClassifier).')
        self.parser.add_argument('--include_normals', type=util.str_to_bool, default=False,
                                 help='Include normal series during training.')
        self.parser.add_argument('--use_hem', type=util.str_to_bool, default=False,
                                 help='If True, use hard example mining in the data loader.')
        self.parser.add_argument('--fine_tune', type=util.str_to_bool, default=True,
                                 help='If True, fine-tune parameters deeper in the network.')
        self.parser.add_argument('--fine_tuning_lr', type=float, default=0.,
                                 help='Initial learning rate for fine-tuning pretrained parameters.')
        self.parser.add_argument('--fine_tuning_boundary', type=str, default='encoders.3',
                                 help='Name of first layer that is not considered a fine-tuning layer.')
