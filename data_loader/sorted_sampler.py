from torch.utils.data.sampler import Sampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler


class SortedSampler(Sampler):
    """SortedSampler is a custom batch sampler for the dataloader which selects
    batch indices under the condition that within a batch, indices must be in sorted
    order."""
    def __init__(self, batch_size, drop_last, data_source, shuffle):
        super(SortedSampler, self).__init__()
        if shuffle:
            self.sampler = RandomSampler(data_source)
        else:
            self.sampler = SequentialSampler(data_source)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(int(idx))
            if len(batch) == self.batch_size:
                batch.sort()
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            batch.sort()
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size
