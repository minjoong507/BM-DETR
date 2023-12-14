import torch
from torch import Tensor
from torch.utils.data import Sampler, BatchSampler, SequentialSampler, RandomSampler
from typing import Iterator, Iterable, Optional, Sequence, List, TypeVar, Generic, Sized, Union


class VideoBatchSampler(Sampler[List[int]]):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler or Iterable): Base sampler. Can be any iterable object
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], batch_size: int, drop_last: bool, dataset: Optional) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        super().__init__(dataset)
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.dataset = dataset

    def __iter__(self) -> Iterator[List[int]]:

        # Implemented based on the benchmarking in https://github.com/pytorch/pytorch/pull/76951
        # if self.drop_last:
        #     sampler_iter = iter(self.sampler)
        #     while True:
        #         try:
        #             batch = [next(sampler_iter) for _ in range(self.batch_size)]
        #             yield batch
        #         except StopIteration:
        #             break
        # else:
        #     batch = [0] * self.batch_size
        #     idx_in_batch = 0
        #     for idx in self.sampler:
        #         batch[idx_in_batch] = idx
        #         idx_in_batch += 1
        #         if idx_in_batch == self.batch_size:
        #             yield batch
        #             idx_in_batch = 0
        #             batch = [0] * self.batch_size
        #     if idx_in_batch > 0:
        #         yield batch[:idx_in_batch]
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            batch_video_ids = []

            for idx in self.sampler:
                vid = self.dataset[idx]['meta']['vid']
                if vid not in batch_video_ids:
                    batch_video_ids.append(vid)
                    batch[idx_in_batch] = idx
                    idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
                    batch_video_ids = []
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


def build_batch_sampler(dataset, bsz):
    batch_sampler = VideoBatchSampler(sampler=RandomSampler(dataset),
                                      batch_size=bsz,
                                      drop_last=False,
                                      dataset=dataset)
    return batch_sampler