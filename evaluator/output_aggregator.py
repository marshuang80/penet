import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import util

from collections import defaultdict


class OutputAggregator(object):
    def __init__(self, agg_method, num_bins=3, num_epochs=5):
        """
        Args:
            agg_method: Name of method used to combine list of outputs into a single output.
               Options are 'max', 'mean'.
            num_bins: Number of bins to use for the histogram if logreg is chosen
            num_epochs: number of epochs to train the logreg model for
        """
        self.agg_method = agg_method
        if self.agg_method == 'max':
            self._reduce = np.max
        elif self.agg_method == 'mean':
            self._reduce = np.mean
        elif self.agg_method == 'logreg':
            self.num_bins = num_bins
            self.num_epochs = num_epochs
            self._reduce = None
        else:
            raise ValueError('Invalid reduce function: {}'.format(agg_method))

        self.classifier = None

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

        if self.agg_method == 'logreg' and phase == 'train':
            self.train_log_reg(key2outputs, data_loader, device)

            # Reduce outputs for each key into a single output
        return {key: self._reduce(outputs) for key, outputs in key2outputs.items()}

    def train_log_reg(self, key2outputs, data_loader, device):
        """Trains the logistic regression reducer and creates and stores the reduce fn."""
        # Reset the model parameters every epoch
        self.classifier = nn.Linear(self.num_bins, 1).float().to(device)

        # Binary classification, for now >:P
        optimizer = optim.Adam(self.classifier.parameters(), lr=.001)
        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(self.num_epochs):
            # Iterate through all series
            for key in key2outputs:
                probs = key2outputs[key]
                hist, _ = np.histogram(probs, bins=self.num_bins, range=(0, 1))

                inputs = torch.tensor(hist, dtype=torch.float32).to(device)
                label = torch.tensor([data_loader.get_series_label(key)], dtype=torch.float32).to(device)
                logits = self.classifier(inputs)
                loss = loss_fn(logits, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        def log_reg_reduce(outputs):
            hist_, _ = np.histogram(outputs, bins=self.num_bins, range=(0, 1))
            inputs_ = torch.tensor(hist_, dtype=torch.float32).to(device)
            return F.sigmoid(self.classifier(inputs_)).detach().cpu().numpy()

        self._reduce = log_reg_reduce
