from torch.utils.data import Dataset
from numpy import random


class SimpleDataLoader:
    r"""
    Data loader for datasets that can yield data immediately and therefore can be loaded sequentially (e.g. a
    TensorDataset that is already placed on the final device).
    Provides an iterable over the given dataset.

    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
    """
    dataset: Dataset
    batch_size: int
    drop_last: bool
    shuffle: bool

    def __init__(self, dataset: Dataset, batch_size: int = 1,
                 shuffle: bool = False, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self) -> iter:

        indices = list(range(len(self.dataset)))  # type: ignore
        if self.shuffle:
            random.shuffle(indices)

        for batch_index in range(len(self.dataset) // self.batch_size):  # type: ignore
            dataset_indices = indices[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
            yield self.dataset[dataset_indices]

        remainder = len(self.dataset) % self.batch_size  # type: ignore[arg-type]
        if not self.drop_last and remainder != 0:
            yield self.dataset[-remainder:]

    def __len__(self) -> int:
        length = len(self.dataset)  # type: ignore
        if self.batch_size is not None:
            from math import ceil
            if self.drop_last:
                length = length // self.batch_size
            else:
                length = ceil(length / self.batch_size)
        return length


if __name__ == '__main__':
    from torch.utils.data.dataset import TensorDataset
    import torch

    data = TensorDataset(torch.randn(501))
    loader = SimpleDataLoader(data, shuffle=True, batch_size=100, drop_last=False)

    for batch in loader:
        print(batch[0].shape)

    print(len(loader))
