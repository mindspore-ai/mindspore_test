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
from enum import Enum
import multiprocessing.context
import numbers
import os
from typing import Any, AnyStr, Callable, Generic, Iterable, List, Mapping, Optional, overload, Protocol, Self, \
    Sequence, TypeVar, Union

import numpy as np
from mindspore import log as logger
from mindspore.common import Tensor
from mindspore.common.generator import Generator

from .dataset import Dataset, IterableDataset
from .sampler import BatachSampler, RandomSampler, Sampler, SequentialSampler
from .utils.fetch import _MapDatasetFetcher, _IterableDatasetFetcher

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_K = TypeVar('_K')
_V = TypeVar('_V')


class _CollateFnType(Protocol):
    @overload
    def __call__(self, batch: Sequence[Tensor]) -> Tensor: ...

    @overload
    def __call__(self, batch: Sequence[np._typing.NDArray[Any]] | Sequence[numbers.Number]) -> np._typing.NDArray[
        Any]: ...

    @overload
    def __call__(self, batch: Sequence[Mapping[_K, _V]]) -> Mapping[_K, _V]: ...

    @overload
    def __call__(self, batch: Sequence[AnyStr]) -> AnyStr: ...

    @overload
    def __call__(self, batch: Sequence[Sequence[_T]]) -> Sequence[_T]: ...


class DatasetType(str, Enum):
    MapDataset: str = "MapDataset"
    IterableDataset: str = "IterableDataset"

class FetcherFactory:
    @staticmethod
    def create_fetcher(dataset_type, dataset, auto_collation, drop_last=False):
        if dataset_type == DatasetType.MapDataset:
            return _MapDatasetFetcher(dataset, auto_collation)
        elif dataset_type == DatasetType.IterableDataset:
            return _IterableDatasetFetcher(dataset, auto_collation, drop_last)
        else:
            raise ValueError("Unknown dataset type: {}".format(dataset_types))

class DataLoader(Generic[_T_co]):
    def __init__(self,
                 dataset: Dataset[_T_co],
                 batch_size: Union[int, None] = 1,
                 shuffle: Union[bool, None] = None,
                 sampler: Union[Sampler, Iterable, None] = None,
                 batch_sampler: Union[Sampler[List], Iterable[List], None] = None,
                 num_workers: int = 0,
                 collate_fn: Union[_CollateFnType, None] = None,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0.,
                 worker_init_fn: Union[Callable[[int], None], None] = None,
                 multiprocessing_context: Union[multiprocessing.context.BaseContext, str, None] = None,
                 generator: Union[Generator, None] = None,
                 *,
                 prefetch_factor: Union[int, None] = None,
                 presistent_workers: bool = False,
                 pin_memory_device: str = "",
                 ) -> None:
        self.dataset = dataset
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.presistent_workers = presistent_workers
        self.generator = generator

        if isinstance(dataset, IterableDataset):
            self.dataset_type = DatasetType.IterableDataset
        else:
            self.dataset_type = DatasetType.MapDataset

        if shuffle is not None and sampler is not None:
            raise ValueError("`shuffle` and `sampler` can not specify at the same time.")
        elif shuffle is None and sampler is None:
            sampler = SequentialSampler(self.dataset)
        elif shuffle is not None:
            sampler = RandomSampler(self.dataset, generator=self.generator)

        self.sampler = sampler

        if batch_sampler is not None and (batch_size is not None or drop_last is not None or sampler is not None):
            raise ValueError("`batch_sampler` can not specify with `batch_size`, `drop_last`, `shuffle` or `sampler`.")

        if batch_sampler is not None:
            self.batch_sampler = batch_sampler
            self.auto_collation = True
        elif batch_size is not None:
            self.batch_sampler = BatachSampler(sampler, batch_size, self.drop_last)
            self.auto_collation = True
        else:
            self.auto_collation = False

        if self.auto_collation:
            self.index_sampler = self.batch_sampler
        else:
            self.index_sampler = self.sampler

    def __iter__(self):
        if self.num_workers > 0:
            if self.presistent_workers:
                if hasattr(self, "_iterator"):
                    self._iterator._reset(self)
                else:
                    self._iterator = _MultiProcessIterator(self)
                return self._iterator
            else:
                return _MultiProcessIterator(self)
        else:
            return _SingleProcessIterator(self)


class _Iterator(Generic[_T_co]):
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataset = dataloader.dataset
        self.drop_last = dataloader.drop_last
        self.dataset_type = dataloader.dataset_type
        self.num_workers = dataloader.num_workers
        self.auto_collation = dataloader.auto_collation
        self.index_sampler = dataloader.index_sampler
        self.sampler_iterator = iter(self.index_sampler)

    def __iter__(self) -> Self:
        return self

    def __len__(self) -> int:
        return len(self.index_sampler)

    def __next__(self) -> Any:
        return self._get_next_data()

    def _get_next_data(self):
        raise NotImplementedError("{} should implement `_get_next_data` method.".format(self.__class__.__name__))

    def _get_next_index(self):
        return next(self.sampler_iterator)

class _SingleProcessIterator(_Iterator):
    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__(dataloader)
        self.dataset_fetcher = FetcherFactory.create_fetcher(self.dataset_type, self.dataset, self.auto_collation,
                                                             self.drop_last)

    def _get_next_data(self):
        next_index = self._get_next_index()
        return self.dataset_fetcher.fetch(next_index)


class _MultiProcessIterator(_Iterator):
    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__(dataloader)
        self._check_num_workers()


    def _check_num_workers(self):
        if hasattr(os, "sched_getaffinity"):
            get_affinity = True
            max_num_workers = len(os.sched_getaffinity(0))
        else:
            get_affinity = False
            max_num_workers = os.cpu_count()

        if self.num_workers > max_num_workers:
            if get_affinity:
                cpu_info = "CPUs {} in the CPU set the current process is restricted to".format(max_num_workers)
            else:
                cpu_info = "logical CPUs {} in the system".format(max_num_workers)
            logger.warning(
                "DataLoader's `num_workers` with value {} is set too high, exceeding the number of {}, which may lead "
                "to competition for resources and slow down performance of DataLoader. It is recommended to reduce the "
                "value of `num_workers`.".format(self.num_workers, cpu_info))

    def _get_next_data(self):
        return 0