import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import util

from collections import defaultdict
from models import AggNet


class OutputAggregator(object):
    def __init__(self, agg_method, num_epochs=5, batch_size=16):
        """
        Args:
            agg_method: Name of method used to combine list of outputs into a single output.
               Options are 'max', 'mean', 'trainable'.
            num_bins: Number of bins to use for the histogram if logreg is chosen
            num_epochs: number of epochs to train the aggregator model for
            batch_size: Batch size for trainable aggregator.
        """
        self.agg_method = agg_method
        self.batch_size = batch_size
        if self.agg_method == 'max':
            self._reduce = np.max
        elif self.agg_method == 'mean':
            self._reduce = np.mean
        elif self.agg_method == 'trainable':
            self.num_epochs = num_epochs
            self._reduce = None
        else:
            raise ValueError('Invalid reduce function: {}'.format(agg_method))

        self.trained_aggregator = None

    def aggregate(self, keys, outputs, data_loader, phase, device):
        """Aggregate model outputs into groups using keys.

        Args:
            keys: List of keys, used for grouping outputs. Must be parallel to outputs.
            outputs: List of outputs to group by key and reduce.
            phase: Phase being aggregated. One of 'train', 'val', 'test'.
            data_loader: DataLoader to sample from
            device: Device on which to run the model.

        Returns:
            Dictionary mapping each series index in idxs to a single scalar probability.
        """
        # Group outputs by key
        key2outputs = defaultdict(list)
        for key, output in zip(keys, outputs):
            key2outputs[key].append(output)

        if self.agg_method == 'trainable' and phase == 'train':
            self.train_aggregator(key2outputs, data_loader, device)

            # Reduce outputs for each key into a single output
        return {key: self._reduce(outputs) for key, outputs in key2outputs.items()}

    def train_aggregator(self, key2outputs, data_loader, device):
        """Trains the logistic regression reducer and creates and stores the reduce fn."""
        util.print_err('Training aggregator...')

        # Reset the model parameters every epoch
        self.trained_aggregator = AggNet(in_channels=1).to(device)

        # Binary classification
        optimizer = optim.Adam(self.trained_aggregator.parameters())
        loss_fn = nn.BCEWithLogitsLoss().to(device)

        examples = [(k, v) for k, v in key2outputs.items()]
        for epoch in range(self.num_epochs):
            # Shuffle and create batches
            random.shuffle(examples)
            batches = [examples[i: i + self.batch_size] for i in range(0, len(examples), self.batch_size)]

            # Iterate through all series
            for batch in batches:
                if len(batch) == 0:
                    break
                keys = [k for k, _ in batch]
                probs = [v for _, v in batch]
                max_len = max(len(p) for p in probs)
                probs = [p + [0.] * (max_len - len(p)) for p in probs]
                labels = [data_loader.get_series_label(k) for k in keys]

                inputs = torch.tensor(probs, dtype=torch.float32).unsqueeze(1).to(device)
                labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1).to(device)
                logits = self.trained_aggregator(inputs)
                loss = loss_fn(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def trained_aggregator_fn(model_outputs):
            if len(model_outputs) < 5:
                return np.array([0.])
            inputs_ = torch.tensor(model_outputs, dtype=torch.float32).view(1, 1, -1).to(device)
            outputs = torch.sigmoid(self.trained_aggregator(inputs_)).detach().cpu().numpy().squeeze()
            return outputs

        self._reduce = trained_aggregator_fn
