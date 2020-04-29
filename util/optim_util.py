import math
import numpy as np
import torch.nn as nn
import torch.optim as optim

from functools import partial
from models.loss import *


def get_loss_fn(is_classification, dataset, size_average=True):
    """Get a loss function to evaluate a model.

    Args:
        is_classification: If true, get loss function for classification.
        dataset: Dataset class name. E.g. 'KineticsDataset'.
        size_average: If True, take mean of outputs (only applies to BinaryFocalLoss).

    Returns:
        Differentiable criterion that can be applied to targets, logits.
    """

    if dataset == 'KineticsDataset':
        return nn.CrossEntropyLoss()
    else:
        return BinaryFocalLoss()


def get_optimizer(parameters, args):
    """Get a PyTorch optimizer for params.

    Args:
        parameters: Iterator of network parameters to optimize (i.e., model.parameters()).
        args: Command line arguments.

    Returns:
        PyTorch optimizer specified by args_.
    """
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, args.learning_rate,
                          momentum=args.sgd_momentum,
                          weight_decay=args.weight_decay,
                          dampening=args.sgd_dampening)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(parameters, args.learning_rate,
                               betas=(args.adam_beta_1, args.adam_beta_2), weight_decay=args.weight_decay)
    else:
        raise ValueError('Unsupported optimizer: {}'.format(args.optimizer))

    return optimizer


def get_scheduler(optimizer, args):
    """Get a learning rate scheduler.

    Args:
        optimizer: The optimizer whose learning rate is modified by the returned scheduler.
        args: Command line arguments.

    Returns:
        PyTorch scheduler that update the learning rate for `optimizer`.
    """
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler == 'multi_step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_decay_gamma)
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=args.lr_decay_gamma, patience=args.patience)
    elif args.lr_scheduler == 'cosine_warmup':
        # If pretrained, delay for warmup steps to allow randomly initialized head to settle down
        ft_lambda_fn = partial(linear_warmup_then_cosine, delay=args.lr_warmup_steps,
                               warmup=args.lr_warmup_steps, max_iter=args.lr_decay_step)
        reg_lambda_fn = partial(linear_warmup_then_cosine, warmup=args.lr_warmup_steps, max_iter=args.lr_decay_step)
        lr_fns = [ft_lambda_fn, reg_lambda_fn] if args.use_pretrained else reg_lambda_fn
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fns)
    else:
        raise ValueError('Invalid learning rate scheduler: {}.'.format(args.lr_scheduler))

    return scheduler


def linear_warmup_then_cosine(last_iter, warmup, max_iter, delay=None):
    if delay is not None:
        last_iter = max(0, last_iter - delay)

    if last_iter < warmup:
        # Linear warmup period
        return float(last_iter) / warmup
    elif last_iter < max_iter:
        # Cosine annealing
        return (1 + math.cos(math.pi * (last_iter - warmup) / max_iter)) / 2
    else:
        # Done
        return 0.


def step_scheduler(lr_scheduler, metrics=None, epoch=None, global_step=None, best_ckpt_metric='val_loss'):
    """Step a LR scheduler."""
    if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
        if best_ckpt_metric in metrics:
            lr_scheduler.step(metrics[best_ckpt_metric], epoch=epoch)
    elif isinstance(lr_scheduler, optim.lr_scheduler.LambdaLR):
        # LambdaLR gets stepped once per iteration
        if epoch is None:
            lr_scheduler.step(global_step)
    elif global_step is None:
        lr_scheduler.step(epoch)


def get_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr.append(param_group['lr'])
    return lr


def softmax(x, axis=0):
    """Compute softmax values for each sets of scores in x.

    Args:
        x: NumPy array to take softmax of.
        axis: Axis over which to take softmax.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis)


class HardExampleMiner(object):

    def __init__(self, example_ids, init_loss=0.69315):
        """Initialize HardExampleMiner to the uniform distribution.

        Args:
            example_ids: List of IDs for each example.
            init_loss: Initial loss for each example (defaults to log(2)).
        """
        self.example_ids = example_ids[:]
        self.losses = [init_loss] * len(self.example_ids)
        self.probs = None
        self._recompute_probs()

    def _recompute_probs(self):
        """Re-compute sample probabilities by taking softmax of losses."""
        self.probs = softmax(np.array(self.losses)).tolist()

    def sample(self):
        """Sample an example according a probability distribution that
        weights hard examples more heavily.

        Returns:
            An example ID for an example drawn according to the probability distribution.
        """
        sampled_idx = np.random.choice(len(self.example_ids), p=self.probs)
        example_id = self.example_ids[sampled_idx]

        return example_id

    def update_distribution(self, example_ids, losses):
        """Update the sample distribution with new examples and their corresponding losses.

        Args:
            example_ids: IDs of examples with losses to update.
            losses: Parallel list of loss values for each example.
        """
        # Update losses
        for example_id, loss in zip(example_ids, losses):
            try:
                idx = self.example_ids.index(example_id)
                self.losses[idx] = loss
            except ValueError:
                # Example was not in list yet
                self.example_ids.append(example_id)
                self.losses.append(loss)

        # Recompute probabilities from losses
        self._recompute_probs()
