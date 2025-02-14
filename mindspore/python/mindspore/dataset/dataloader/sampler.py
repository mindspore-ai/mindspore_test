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

import mindspore as ms
import itertools
from typing import Generic, Iterable, Iterator, TypeVar, Union

_T_co = TypeVar("_T_co", covariant=True)


class Sampler(Generic[_T_co]):
    def __init__(self, data_source=None) -> None:
        pass


class SequentialSampler(Sampler):
    def __init__(self, data_source) -> None:
        self.data_source = data_source

    def __iter__(self)->Iterator[int]:
        yield from range(len(self.data_source))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self,
                 data_source,
                 replacement: bool=False,
                 num_samples: Union[int, None] = None,
                 generator=None) -> None:
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
            generator = ms.Generator()
            generator.manual_seed(seed)
        else:
            # no mao yong
            generator = self.generator

        # 有放回
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from ms.ops.randint(low=0, high=n, size=(32,), 
                                          dtype=ms.int64, seed=seed).tolist()
            yield from ms.ops.randint(low=0, high=n, size=(self.num_samples % 32,),
                                      dtype=ms.int64, seed=seed).tolist()
        # 无放回
        else:
            for _ in range(self.num_samples // n):
                yield from ms.ops.randperm(n, seed=seed).tolist()
            yield from ms.ops.randperm(n, seed=seed).tolist()[: self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples


class BatchSampler(Sampler):
    def __init__(self,
                 sampler: Union[Sampler, Iterable],
                 batch_size: int,
                 drop_last: bool) -> None:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(f"batch_size should be a positive integer value, but got batch_size={batch_size}")
        if not isinstance(drop_last, bool):
            raise ValueError(f"drop_last should be a boolean value, but got drop_last={drop_last}")

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
        raise NotImplementedError


class InfiniteSampler(Sampler):
    r"""
    Used as sampler for :class:`~mindspore.dataset.dataloader.IterableDataset`.
    """

    def __iter__(self):
        while True:
            yield None