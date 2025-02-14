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
    def __init__(self, data_source=None) -> None:
        pass


class SequentialSampler(Sampler):
    def __init__(self, data_source) -> None:
        super().__init__(data_source)
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        yield from range(len(self.data_source))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler[int]):
    """
    Sampler that samples elements randomly.

    Args:
        data_source (Dataset): The data source to sample from.
        replacement (bool, optional): Whether to put the element back for the next draw. Default: False.
        num_samples (int, optional): The number of samples to draw. Default: None, set to the length of `data_source`.
        generator (mindspore.Generator, optional): The generator to use in sampling. Default: None, not deterministic.
    """

    def __init__(
            self,
            data_source,
            replacement: bool = False,
            num_samples: Union[int, None] = None,
            generator=None,
    ) -> None:
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = ms.get_seed()
            if seed is not None:
                generator = ms.Generator()
                generator.manual_seed(seed)
        else:
            # no need to use custom generator
            generator = self.generator

        # with replacement
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from ms.ops.randint(
                    low=0, high=n, size=(32,), dtype=ms.int64, seed=seed
                ).tolist()
            yield from ms.ops.randint(
                low=0, high=n, size=(self.num_samples % 32,), dtype=ms.int64, seed=seed
            ).tolist()
        # without replacement
        else:
            for _ in range(self.num_samples // n):
                yield from ms.ops.randperm(n, seed=seed).tolist()
            yield from ms.ops.randperm(n, seed=seed).tolist()[: self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler[list[int]]):
    """
    Sampler that yields a mini-batch of indices at a time.

    Args:
        sampler (Union[Sampler, Iterable]): The base sampler used to yield indices.
        batch_size (int): The size of the mini-batch.
        drop_last (bool): Whether to drop the last batch if it is less than `batch_size`.
    """

    def __init__(self, sampler: Union[Sampler, Iterable], batch_size: int, drop_last: bool) -> None:
        super().__init__()
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise TypeError(f"batch_size must be int, but got: {type(batch_size).__name__}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, but got: {batch_size}")
        if not isinstance(drop_last, bool):
            raise TypeError(f"drop_last must be bool, but got: {type(drop_last).__name__}.")

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
            # auto slicing with itertools
            batch = [*itertools.islice(sampler_iter, self.batch_size)]
            while batch:
                yield batch
                batch = [*itertools.islice(sampler_iter, self.batch_size)]

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) - 1) // self.batch_size + 1


class InfiniteSampler(Sampler):
    r"""
    Used as sampler for :class:`~mindspore.dataset.dataloader.IterableDataset`.
    """

    def __iter__(self):
        while True:
            yield None
