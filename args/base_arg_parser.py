import argparse
import datetime
import json
import numpy as np
import os
import random
import torch
import torch.backends.cudnn as cudnn
import util


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='PENet base args')
        self.parser.add_argument('--model', type=str, choices=('PENet', 'PENetClassifier'), default='PENet',
                                 help='Model to use. PENetClassifier or PENet.')
        self.parser.add_argument('--batch_size', type=int, default=6, help='Batch size.')
        self.parser.add_argument('--ckpt_path', type=str, default='',
                                 help='Path to checkpoint to load. If empty, start from scratch.')
        self.parser.add_argument('--data_dir', type=str, required=True,
                                 help='Path to data directory with both normal and aneurysm studies.')
        self.parser.add_argument('--pkl_path', type=str, default='',
                                 help='Path to pickled CTSeries list. Defaults to data_dir/series_list.pkl')
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2',
                                 help='Comma-separated list of GPU IDs. Use -1 for CPU.')
        self.parser.add_argument('--init_method', type=str, default='kaiming', choices=('kaiming', 'normal', 'xavier'),
                                 help='Initialization method to use for conv kernels and linear weights.')
        self.parser.add_argument('--model_depth', default=50, type=int,
                                 help='Depth of the model. Meaning of depth depends on the model.')
        self.parser.add_argument('--name', type=str, required=True, help='Experiment name.')
        self.parser.add_argument('--img_format', type=str, default='raw', choices=('raw', 'png'),
                                 help='Format for input images: "raw" means raw Hounsfield Units, "png" means PNG.')
        self.parser.add_argument('--resize_shape', type=str, default='224,224',
                                 help='Comma-separated 2D shape for images after resizing (before cropping).')
        self.parser.add_argument('--crop_shape', type=str, default='208,208',
                                 help='Comma-separated 2D shape for images after cropping (crop comes after resize).')
        self.parser.add_argument('--min_abnormal_slices', type=int, default=4,
                                 help='Minimum number of slices with abnormality for window to be considered abnormal.')
        self.parser.add_argument('--num_channels', default=3, type=int, help='Number of channels in an image.')
        self.parser.add_argument('--num_classes', default=1, type=int, help='Number of classes to predict.')
        self.parser.add_argument('--num_slices', default=32, type=int, help='Number of slices to use per study.')
        self.parser.add_argument('--num_visuals', type=int, default=4,
                                 help='Maximum number of visuals per evaluation.')
        self.parser.add_argument('--num_workers', default=8, type=int, help='Number of threads for the DataLoader.')
        self.parser.add_argument('--agg_method', type=str, default='', choices=('max', 'mean', 'logreg', ''),
                                 help='Method for aggregating window-level predictions to a single prediction.')
        self.parser.add_argument('--save_dir', type=str, default='../ckpts/',
                                 help='Directory in which to save model checkpoints.')
        self.parser.add_argument('--threshold_size', default=0, type=float,
                                 help='Only the aneurysms bigger than the threshold size (mm) will be labeled as 1.')
        self.parser.add_argument('--toy', type=util.str_to_bool, default=False, help='Use small dataset or not.')
        self.parser.add_argument('--toy_size', type=int, default=5,
                                 help='How many of each type to include in the toy dataset.')
        self.parser.add_argument('--series', default='sagittal', type=str, choices=('sagittal', 'axial', 'coronal'), 
                                 help='The series to use -- one of (sagittal / axial / coronal')
        self.parser.add_argument('--vstep_size', type=int, default=1, 
                                 help='Number of slices to move forward at a time')
        self.parser.add_argument('--dataset', type=str, required=True,
                                 choices=('kinetics', 'pe'),
                                 help='Dataset to use.')
        self.parser.add_argument('--deterministic', type=util.str_to_bool, default=False,
                                 help='If true, set a random seed to get deterministic results.')
        self.parser.add_argument('--cudnn_benchmark', type=util.str_to_bool, default=False,
                                 help='Set cudnn benchmark to save fastest computation algorithm for fixed size inputs. \
                                       Turn off when input size is variable.')
        self.parser.add_argument('--hide_probability', type=float, default=0.0,
                                 help='Probability of hiding squares in hide-and-seek.')
        self.parser.add_argument('--hide_level', type=str, choices=('window', 'image'), default='window',
                                 help='Level of hiding squares in hide-and-seek.')
        self.parser.add_argument('--only_topmost_window', type=util.str_to_bool, default=False,
                                 help='If true, only use the topmost window in each series.')
        self.parser.add_argument('--eval_mode', type=str, choices=('window', 'series'), default='series',
                                 help='Evaluation mode for reporting metrics.')
        self.parser.add_argument('--do_classify', type=util.str_to_bool, default=False,
                                 help='If True, perform classification.')
        self.parser.add_argument('--pe_types', type=eval, default='["central", "segmental"]',
                                 help='Types of PE to include.')
        self.is_training = None

    def parse_args(self):
        args = self.parser.parse_args()

        # Save args to a JSON file
        date_string = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(args.save_dir, '{}_{}'.format(args.name, date_string))
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')
        args.save_dir = save_dir

        # Add configuration flags outside of the CLI
        args.is_training = self.is_training
        args.start_epoch = 1  # Gets updated if we load a checkpoint
        if not args.is_training and not args.ckpt_path and not (hasattr(args, 'test_2d') and args.test_2d):
            raise ValueError('Must specify --ckpt_path in test mode.')
        if args.is_training and args.epochs_per_save % args.epochs_per_eval != 0:
            raise ValueError('epochs_per_save must be divisible by epochs_per_eval.')
        if args.is_training:
            args.maximize_metric = not args.best_ckpt_metric.endswith('loss')
            if args.lr_scheduler == 'multi_step':
                args.lr_milestones = util.args_to_list(args.lr_milestones, allow_empty=False)
        if not args.pkl_path:
            args.pkl_path = os.path.join(args.data_dir, 'series_list.pkl')

        # Set up resize and crop
        args.resize_shape = util.args_to_list(args.resize_shape, allow_empty=False, arg_type=int, allow_negative=False)
        args.crop_shape = util.args_to_list(args.crop_shape, allow_empty=False, arg_type=int, allow_negative=False)

        # Set up available GPUs
        args.gpu_ids = util.args_to_list(args.gpu_ids, allow_empty=True, arg_type=int, allow_negative=False)
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
            cudnn.benchmark = args.cudnn_benchmark
        else:
            args.device = 'cpu'

        # Set random seed for a deterministic run
        if args.deterministic:
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            cudnn.deterministic = True

        # Map dataset name to a class
        if args.dataset == 'kinetics':
            args.dataset = 'KineticsDataset'
        elif args.dataset == 'pe':
            args.dataset = 'CTPEDataset3d'

        if self.is_training and args.use_pretrained:
            if args.model != 'PENet' and args.model != 'PENetClassifier':
                raise ValueError('Pre-training only supported for PENet/PENetClassifier loading PENetClassifier.')
            if not args.ckpt_path:
                raise ValueError('Must specify a checkpoint path for pre-trained model.')

        args.data_loader = 'CTDataLoader'
        if args.model == 'PENet':
            if args.model_depth != 50:
                raise ValueError('Invalid model depth for PENet: {}'.format(args.model_depth))
            args.loader = 'window'
        elif args.model == 'PENetClassifier':
            if args.model_depth != 50:
                raise ValueError('Invalid model depth for PENet: {}'.format(args.model_depth))
            args.loader = 'window'
            if args.dataset == 'KineticsDataset':
                args.data_loader = 'KineticsDataLoader'

        # Set up output dir (test mode only)
        if not self.is_training:
            args.results_dir = os.path.join(args.results_dir, '{}_{}'.format(args.name, date_string))
            os.makedirs(args.results_dir, exist_ok=True)

        return args
