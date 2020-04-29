import util

from time import time
from .base_logger import BaseLogger


class TrainLogger(BaseLogger):
    """Class for logging training info to the console and saving model parameters to disk."""
    def __init__(self, args, dataset_len, pixel_dict):
        super(TrainLogger, self).__init__(args, dataset_len, pixel_dict)

        assert args.is_training
        assert args.iters_per_print % args.batch_size == 0, "iters_per_print must be divisible by batch_size"
        assert args.iters_per_visual % args.batch_size == 0, "iters_per_visual must be divisible by batch_size"

        self.iters_per_print = args.iters_per_print
        self.iters_per_visual = args.iters_per_visual
        self.experiment_name = args.name
        self.max_eval = args.max_eval
        self.num_epochs = args.num_epochs
        self.loss_meters = self._init_loss_meters()
        self.loss_meter = util.AverageMeter()

    def _init_loss_meters(self):
        loss_meters = {}

        if self.do_classify:
            loss_meters['cls_loss'] = util.AverageMeter()

        return loss_meters

    def _reset_loss_meters(self):
        for v in self.loss_meters.values():
            v.reset()

    def _update_loss_meters(self, n, cls_loss=None, seg_loss=None):
        if cls_loss is not None:
            self.loss_meters['cls_loss'].update(cls_loss, n)
        if seg_loss is not None:
            self.loss_meters['seg_loss'].update(seg_loss, n)

    def _get_avg_losses(self, as_string=False):
        if as_string:
            s = ', '.join('{}: {:.3g}'.format(k, v.avg) for k, v in self.loss_meters.items())
            return s
        else:
            loss_dict = {'batch_{}'.format(k): v.avg for k, v in self.loss_meters.items()}
            return loss_dict

    def start_iter(self):
        """Log info for start of an iteration."""
        self.iter_start_time = time()

    def log_iter(self, inputs, cls_logits, targets, cls_loss, optimizer):
        """Log results from a training iteration."""
        cls_loss = None if cls_loss is None else cls_loss.item()
        self._update_loss_meters(inputs.size(0), cls_loss)

        # Periodically write to the log and TensorBoard
        if self.iter % self.iters_per_print == 0:

            # Write a header for the log entry
            avg_time = (time() - self.iter_start_time) / self.batch_size
            message = '[epoch: {}, iter: {} / {}, time: {:.2f}, {}]' \
                .format(self.epoch, self.iter, self.dataset_len, avg_time, self._get_avg_losses(as_string=True))

            # Write all errors as scalars to the graph
            scalar_dict = self._get_avg_losses()
            scalar_dict.update({'train/lr{}'.format(i): pg['lr'] for i, pg in enumerate(optimizer.param_groups)})
            self._log_scalars(scalar_dict, print_to_stdout=False)
            self._reset_loss_meters()

            self.write(message)

        # Periodically visualize up to num_visuals training examples from the batch
        if self.iter % self.iters_per_visual == 0:
            self.visualize(inputs, cls_logits, targets, phase='train')

    def end_iter(self):
        """Log info for end of an iteration."""
        self.iter += self.batch_size
        self.global_step += self.batch_size

    def start_epoch(self):
        """Log info for start of an epoch."""
        self.epoch_start_time = time()
        self.iter = 0
        self.write('[start of epoch {}]'.format(self.epoch))

    def end_epoch(self, metrics, curves):
        """Log info for end of an epoch.

        Args:
            metrics: Dictionary of metric values. Items have format '{phase}_{metric}': value.
            curves: Dictionary of curves. Items have format '{phase}_{curve}: value.
        """
        self.write('[end of epoch {}, epoch time: {:.2g}]'.format(self.epoch, time() - self.epoch_start_time))
        self._log_scalars(metrics)

        self._plot_curves(curves)

        self.epoch += 1

    def is_finished_training(self):
        """Return True if finished training, otherwise return False."""
        return 0 < self.num_epochs < self.epoch
