import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F
import util

from datetime import datetime
from tensorboardX import SummaryWriter

plt.switch_backend('agg')


class BaseLogger(object):
    def __init__(self, args, dataset_len, pixel_dict):

        def round_down(x, m):
            """Round x down to a multiple of m."""
            return int(m * round(float(x) / m))

        self.args = args
        self.batch_size = args.batch_size
        self.dataset_len = dataset_len
        self.device = args.device
        self.img_format = args.img_format
        self.save_dir = args.save_dir if args.is_training else args.results_dir
        self.do_classify = args.do_classify
        self.num_visuals = args.num_visuals
        self.log_path = os.path.join(self.save_dir, '{}.log'.format(args.name))
        log_dir = os.path.join('logs', args.name + '_' + datetime.now().strftime('%b%d_%H%M'))
        self.summary_writer = SummaryWriter(log_dir=log_dir)

        self.epoch = args.start_epoch
        # Current iteration in epoch (i.e., # examples seen in the current epoch)
        self.iter = 0
        # Current iteration overall (i.e., total # of examples seen)
        self.global_step = round_down((self.epoch - 1) * dataset_len, args.batch_size)
        self.iter_start_time = None
        self.epoch_start_time = None
        self.pixel_dict = pixel_dict

    def _log_scalars(self, scalar_dict, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.write('[{}: {:.3g}]'.format(k, v))
            k = k.replace('_', '/')  # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, self.global_step)

    def _plot_curves(self, curves_dict):
        """Plot all curves in a dict as RGB images to TensorBoard."""
        for name, curve in curves_dict.items():
            fig = plt.figure()
            ax = plt.gca()

            plot_type = name.split('_')[-1]
            ax.set_title(plot_type)
            if plot_type == 'PRC':
                precision, recall, _ = curve
                ax.step(recall, precision, color='b', alpha=0.2, where='post')
                ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
            elif plot_type == 'ROC':
                false_positive_rate, true_positive_rate, _ = curve
                ax.plot(false_positive_rate, true_positive_rate, color='b')
                ax.plot([0, 1], [0, 1], 'r--')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
            else:
                ax.plot(curve[0], curve[1], color='b')

            ax.set_ylim([0.0, 1.05])
            ax.set_xlim([0.0, 1.0])

            fig.canvas.draw()

            curve_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            curve_img = curve_img.reshape((3,) + fig.canvas.get_width_height()[::-1])
            self.summary_writer.add_image(name.replace('_', '/'), curve_img, global_step=self.global_step)

    def visualize(self, inputs, cls_logits, targets_dict, phase, unique_id=None):
        """Visualize predictions and targets in TensorBoard.

        Args:
            inputs: Inputs to the model.
            cls_logits: Classification logits predicted by the model.
            targets_dict: Dictionary of information about the target labels.
            phase: One of 'train', 'val', or 'test'.
            unique_id: A unique ID to append to every image title. Allows
              for displaying all visualizations separately on TensorBoard.

        Returns:
            Number of examples visualized to TensorBoard.
        """

        if self.pixel_dict is None:
            # Set pixel_dict to None to bypass visualization
            return 0

        cls_logits = cls_logits.detach().to('cpu')

        cls_probs = F.sigmoid(cls_logits).numpy()

        is_3d = inputs.dim() > 4
        num_visualized = 0
        for i in range(self.num_visuals):
            if i >= inputs.shape[0]:
                break

            input_np = util.un_normalize(inputs[i], self.img_format, self.pixel_dict)
            input_np = input_np.astype(np.float32) / 255.

            mask_np = None
            output_np = None

            label = 'abnormal' if targets_dict['is_abnormal'][i] else 'normal'
            visuals_np = input_np
            title = 'input'

            tag = '{}/{}/{}_{}_{:.4f}'.format(phase, title, label, targets_dict['dset_path'][i], cls_probs[i][0])
            if unique_id is not None:
                tag += '_{}'.format(unique_id)
            
            # Reshaping to B, C, T, H, W
            visuals_np = np.expand_dims(visuals_np, 0)
            if is_3d:
                self.summary_writer.add_video(tag, visuals_np, self.global_step)
            else:
                self.summary_writer.add_image(tag, visuals_np, self.global_step)

            num_visualized += 1

        return num_visualized

    def write(self, message, print_to_stdout=True):
        """Write a message to the log. If print_to_stdout is True, also print to stdout."""
        with open(self.log_path, 'a') as log_file:
            log_file.write(message + '\n')
        if print_to_stdout:
            print(message)

    def start_iter(self):
        """Log info for start of an iteration."""
        raise NotImplementedError

    def end_iter(self):
        """Log info for end of an iteration."""
        raise NotImplementedError

    def start_epoch(self):
        """Log info for start of an epoch."""
        raise NotImplementedError

    def end_epoch(self, metrics, curves):
        """Log info for end of an epoch. Save model parameters and update learning rate."""
        raise NotImplementedError
