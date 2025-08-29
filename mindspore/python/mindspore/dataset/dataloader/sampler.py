# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sampler module."""

import itertools
from typing import Generic, Iterable, Iterator, TypeVar, Union

import mindspore as ms


_T_co = TypeVar("_T_co", covariant=True)


class Sampler(Generic[_T_co]):
    """
    Base Class of the Sampler

    Args:
        data_source (Dataset, optional): Dataset to be sampled. Default: ``None`` .
    """
    def __init__(self, data_source=None) -> None:
        pass


class SequentialSampler(Sampler):
    """
    Samples the dataset elements sequentially.

    Args:
        data_source (Dataset): Dataset to be sampled.

    Examples:
        >>> from mindspore.dataset.dataloader import SequentialSampler
        >>>
        >>> dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> sampler = SequentialSampler(dataset)
    """
    def __init__(self, data_source) -> None:
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        yield from range(len(self.data_source))

    def __len__(self) -> int:
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """
    Samples the dataset elements randomly.

    Args:
        data_source (Dataset): Dataset to be sampled.
        replacement (bool, optional): Whether to enable the return sampling. Default: ``False`` .
        num_samples (int, optional): Number of samples to be drawn. Default: ``None`` ,
            will be set to the length of `data_source` .
        generator (mindspore.Generator, optional): Generator used during sampling. Default: ``None`` .

    Examples:
        >>> from mindspore.dataset.dataloader import RandomSampler
        >>>
        >>> dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> sampler = RandomSampler(dataset)
    """

    def __init__(
            self,
            data_source,
            replacement: bool = False,
            num_samples: Union[int, None] = None,
            generator=None,
    ) -> None:
        super().__init__(data_source)
        if not isinstance(replacement, bool):
            raise TypeError(f"replacement must be bool, but got: {type(replacement).__name__}")
        if num_samples is not None and not isinstance(num_samples, int):
            raise TypeError(f"num_samples must be int, but got: {type(num_samples).__name__}")
        if num_samples is not None and num_samples <= 0:
            raise ValueError(f"num_samples must be a positive integer value, but got num_samples = {num_samples}")
        if generator is not None and not isinstance(generator, ms.Generator):
            raise TypeError(f"generator must be mindspore.Generator, but got: {type(generator).__name__}")
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

    @property
    def num_samples(self) -> int:
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        seed = ms.get_seed()
        if self.generator is None:
            if seed is not None:
                generator = ms.Generator()
                generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from ms.ops.randint(
                    low=0, high=n, size=(32,), dtype=ms.int64, seed=seed
                ).tolist()
            yield from ms.ops.randint(
                low=0, high=n, size=(self.num_samples % 32,), dtype=ms.int64, seed=seed
            ).tolist()
        else:
            if seed is None:
                seed = -1
            for _ in range(self.num_samples // n):
                yield from ms.ops.randperm(n, seed=seed).tolist()
            yield from ms.ops.randperm(n, seed=seed).tolist()[: self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[list[int]]):
    """
   A sampler that generates mini-batch indices each time.

    Args:
        sampler (Union[Sampler, Iterable]): Sampler for generating indices.
        batch_size (int): The size of the mini batch.
        drop_last (bool): Whether to discard the last batch of data if the batch is smaller than `batch_size` .

    Examples:
        >>> from mindspore.dataset.dataloader import BatchSampler, SequentialSampler
        >>>
        >>> dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> sequential_sampler = SequentialSampler(dataset)
        >>> batch_sampler = BatchSampler(sequential_sampler, 2, False)
    """

    def __init__(self, sampler: Union[Sampler, Iterable], batch_size: int, drop_last: bool) -> None:
        super().__init__()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError(f"batch_size must be int, but got: {type(batch_size).__name__}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer value, but got batch_size = {batch_size}")
        if not isinstance(drop_last, bool):
            raise TypeError(f"drop_last must be bool, but got: {type(drop_last).__name__}")

        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[list[int]]:
        sampler_iter = iter(self.sampler)
        if self.drop_last:
            # Create multiple references to the same iterator
            args = [sampler_iter] * self.batch_size
            # zip will call elements of args in sequence, equals to call generator batch-size times
            for batch_droplast in zip(*args):
                yield [*batch_droplast]
        else:
            batch = [*itertools.islice(sampler_iter, self.batch_size)]
            while batch:
                yield batch
                batch = [*itertools.islice(sampler_iter, self.batch_size)]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) - 1) // self.batch_size + 1


class InfiniteSampler(Sampler):
    """
    Used as sampler for :class:`~mindspore.dataset.dataloader.IterableDataset`.
    """

    def __iter__(self):
        while True:
            yield None
