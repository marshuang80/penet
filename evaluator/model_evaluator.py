import numpy as np
import random
import sklearn.metrics as sk_metrics
import torch
import torch.nn.functional as F
import util

from tqdm import tqdm
from .output_aggregator import OutputAggregator


class ModelEvaluator(object):
    """Class for evaluating a model during training."""

    def __init__(self, do_classify, dataset_name, data_loaders, logger,
                 agg_method=None, num_visuals=None, max_eval=None, epochs_per_eval=1):
        """
        Args:
            do_classify: If True, evaluate classification metrics.
            dataset_name: Name of dataset class.
            data_loaders: List of Torch `DataLoader`s to sample from.
            logger: Logger for plotting to console and TensorBoard.
            agg_method: Method used to aggregate outputs. None, 'max', 'mean', or 'logreg'.
            num_visuals: Number of visuals to display.
            max_eval: Maximum number of examples to evaluate at each evaluation.
            epochs_per_eval: Number of epochs between each evaluation.
        """
        self.aggregator = None if not agg_method else OutputAggregator(agg_method, num_bins=10, num_epochs=5)
        self.data_loaders = data_loaders
        self.dataset_name = dataset_name
        self.do_classify = do_classify
        self.epochs_per_eval = epochs_per_eval
        self.logger = logger
        self.cls_loss_fn = None if not do_classify else util.optim_util.get_loss_fn(is_classification=True, dataset=dataset_name)
        self.seg_loss_fn = util.optim_util.get_loss_fn(is_classification=False, dataset=dataset_name)
        self.num_visuals = num_visuals
        self.max_eval = None if max_eval is None or max_eval < 0 else max_eval

    def evaluate(self, model, device, epoch=None):
        """Evaluate a model at the end of the given epoch.

        Args:
            model: Model to evaluate.
            device: Device on which to evaluate the model.
            epoch: The epoch that just finished. Determines whether to evaluate the model.
        Returns:
            metrics: Dictionary of metrics for the current model.
            curves: Dictionary of curves for the current model. E.g. ROC.
        Notes:
            Returned dictionaries will be empty if not an evaluation epoch.
        """
        metrics, curves = {}, {}

        if epoch is None or epoch % self.epochs_per_eval == 0:
            # Evaluate on the training and validation sets
            model.eval()
            for data_loader in self.data_loaders:
                phase_metrics, phase_curves = self._eval_phase(model, data_loader, data_loader.phase, device)
                metrics.update(phase_metrics)
                curves.update(phase_curves)
            model.train()

        return metrics, curves

    def _eval_phase(self, model, data_loader, phase, device):
        """Evaluate a model for a single phase.

        Args:
            model: Model to evaluate.
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.
        """
        batch_size = data_loader.batch_size_

        # Keep track of task-specific records needed for computing overall metrics
        if self.aggregator is not None:
            records = {'keys': [], 'probs': []}
        else:
            records = {'loss_meter': util.AverageMeter()}

        num_examples = len(data_loader.dataset)
        if self.max_eval is not None:
            num_examples = min(num_examples, self.max_eval)

        # Sample from the data loader and record model outputs
        num_evaluated = num_visualized = 0
        start_visual = random.randint(0, max(1, num_examples - self.num_visuals))
        with tqdm(total=num_examples, unit=' ' + phase) as progress_bar:
            for inputs, targets_dict in data_loader:
                if num_evaluated >= num_examples:
                    break

                with torch.no_grad():
                    cls_logits = model.forward(inputs.to(device))
                    cls_targets = targets_dict['is_abnormal']
                    loss = self.cls_loss_fn(cls_logits, cls_targets.to(device))

                self._record_batch(cls_logits, targets_dict['series_idx'], loss, **records)

                if start_visual <= num_evaluated and num_visualized < self.num_visuals and phase != 'train':
                    num_visualized += self.logger.visualize(inputs, cls_logits, targets_dict, phase=phase)

                progress_bar.update(min(batch_size, num_examples - num_evaluated))
                num_evaluated += batch_size

        # Map to summary dictionaries
        metrics, curves = self._get_summary_dicts(data_loader, phase, device, **records)

        return metrics, curves

    @staticmethod
    def _record_batch(logits, targets, loss, probs=None, keys=None, loss_meter=None):
        """Record results from a batch to keep track of metrics during evaluation.

        Args:
            logits: Batch of logits output by the model.
            targets: Batch of ground-truth targets corresponding to the logits.
            probs: List of probs from all evaluations.
            keys: List of keys to map window-level logits back to their series-level predictions.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        """
        if probs is not None:
            assert keys is not None, 'Must keep probs and keys lists in parallel'
            with torch.no_grad():
                batch_probs = F.sigmoid(logits)
            probs.append(batch_probs)

            # Note: `targets` is assumed to hold the keys for these examples
            keys.append(targets)

        if loss_meter is not None:
            loss_meter.update(loss.item(), logits.size(0))

    def _get_summary_dicts(self, data_loader, phase, device, probs=None, keys=None, loss_meter=None):
        """Get summary dictionaries given dictionary of records kept during evaluation.

        Args:
            data_loader: Torch DataLoader to sample from.
            phase: Phase being evaluated. One of 'train', 'val', or 'test'.
            device: Device on which to evaluate the model.
            probs: List of probs from all evaluations.
            keys: List of keys to map window-level logits back to their series-level predictions.
            loss_meter: AverageMeter keeping track of average loss during evaluation.
        Returns:
            metrics: Dictionary of metrics for the current model.
            curves: Dictionary of curves for the current model. E.g. ROC.
        """
        metrics, curves = {}, {}

        if probs is not None:
            # If records kept track of individual probs and keys, implied that we need to aggregate them
            assert keys is not None, 'Must keep probs and keys lists in parallel.'
            assert self.aggregator is not None, 'Must specify an aggregator to aggregate probs and keys.'

            # Convert to flat numpy array
            probs = np.concatenate(probs).ravel().tolist()
            keys = np.concatenate(keys).ravel().tolist()

            # Aggregate predictions across each series
            idx2prob = self.aggregator.aggregate(keys, probs, data_loader, phase, device)
            probs, labels = [], []
            for idx, prob in idx2prob.items():
                probs.append(prob)
                labels.append(data_loader.get_series_label(idx))
            probs, labels = np.array(probs), np.array(labels)

            # Update summary dicts
            metrics.update({
                phase + '_' + 'loss': sk_metrics.log_loss(labels, probs, labels=[0, 1])
            })

            # Update summary dicts
            try:
                metrics.update({
                    phase + '_' + 'AUPRC': sk_metrics.average_precision_score(labels, probs),
                    phase + '_' + 'AUROC': sk_metrics.roc_auc_score(labels, probs),
                })
                curves.update({
                    phase + '_' + 'PRC': sk_metrics.precision_recall_curve(labels, probs),
                    phase + '_' + 'ROC': sk_metrics.roc_curve(labels, probs)
                })
            except ValueError:
                pass

        if loss_meter is not None:
            metrics.update({
                phase + '_' + 'loss': loss_meter.avg
            })

        return metrics, curves
