import datasets
import torch

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from .padded_inputs import PaddedInputs
from .sorted_sampler import SortedSampler


class CTDataLoader(DataLoader):
    """ Base class DataLoader for loading a 3d dataset. This data loader is designed to work with
    sequential models, and takes care of sorting batches and padding them for the pytorch
    recurrent networks. Note that the dataset MUST BE SORTED BY LENGTH for this to work."""
    def __init__(self, args, phase, is_training=True):
        dataset_fn = datasets.__dict__[args.dataset]
        dataset = dataset_fn(args, phase, is_training)
        self.batch_size_ = args.batch_size
        self.phase = phase
        if args.loader == 'series':
            batch_sampler = SortedSampler(batch_size=args.batch_size,
                                          drop_last=True,
                                          data_source=dataset,
                                          shuffle=is_training)
            super(CTDataLoader, self).__init__(dataset,
                                               num_workers=args.num_workers,
                                               batch_sampler=batch_sampler,
                                               collate_fn=self.pad_sequences,
                                               pin_memory=True)
        elif args.loader == 'window' or args.loader == 'slice':
            super(CTDataLoader, self).__init__(dataset,
                                               batch_size=args.batch_size,
                                               shuffle=is_training,
                                               num_workers=args.num_workers,
                                               pin_memory=True)
        else:
            raise NotImplementedError('Invalid args.loader: {}'.format(args.loader))

    def get_series_label(self, series_idx):
        """Get a floating point label for a series at given index."""
        return self.dataset.get_series_label(series_idx)

    @staticmethod
    def pad_sequences(batch):
        """Provides batching for the data loader by padding sequences and stacking them
        into a padded tensor.

        Args:
            batch: List of tensors of shape channels x seq_length x height x width.

        Returns: PaddedInputs object containing the padded sequences and their lengths,
            along with the labels.
        """
        data_batch = [slice_[0] for slice_ in batch]
        seq_lengths = [slice_.shape[1] for slice_ in data_batch]
        seq_lengths = torch.tensor(seq_lengths, dtype=torch.int64)
        target = [item[1] for item in batch]
        target = torch.tensor(target, dtype=torch.float32).unsqueeze(1)
        padded_batch = pad_sequence(data_batch, batch_first=True)
        output = PaddedInputs(padded_batch, seq_lengths)

        return output, target

    def get_series(self, study_num):
        """Get a series with given dset_path. Note: Slow function, avoid this in training."""
        return self.dataset.get_series(study_num)
