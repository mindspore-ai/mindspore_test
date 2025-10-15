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
"""
Dataloader module.
"""

from enum import Enum
import multiprocessing
import numbers
import os
import queue
import threading
from typing import Any, AnyStr, Callable, Generic, Iterable, List, Mapping, overload, Protocol, Sequence, TypeVar, Union

import numpy as np
import numpy.typing as npt

import mindspore as ms
import mindspore._c_dataengine as cde
from mindspore import log as logger
from mindspore.common import Tensor
from .dataset import Dataset, IterableDataset
from .sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler, InfiniteSampler
from ._utils import WORKER_TIME_OUT
from ._utils.fetch import MapDatasetFetcher, IterableDatasetFetcher
from ._utils.pin_memory import pin_memory_fn, pin_worker_fn
from ._utils.signal_handling import set_sigchld_handler
from ._utils.worker import data_worker_fn, WorkerException, ResumeIterationFlag

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_K = TypeVar('_K')
_V = TypeVar('_V')


class _CollateFnType(Protocol):
    """
    Protocol for the collate function.
    """

    @overload
    def __call__(self, batch: Union[Sequence[npt.NDArray], Sequence[numbers.Number]]) -> npt.NDArray:
        pass

    @overload
    def __call__(self, batch: Sequence[Tensor]) -> Tensor:  # pylint: disable=function-redefined
        pass

    @overload
    def __call__(self, batch: Sequence[Mapping[_K, _V]]) -> Mapping[_K, _V]:  # pylint: disable=function-redefined
        pass

    @overload
    def __call__(self, batch: Sequence[AnyStr]) -> AnyStr:  # pylint: disable=function-redefined
        pass

    @overload
    def __call__(self, batch: Sequence[Sequence[_T]]) -> Sequence[_T]:  # pylint: disable=function-redefined
        pass


class DatasetType(str, Enum):
    """
    Enum for the dataset type.
    """

    MapDataset: str = "MapDataset"
    IterableDataset: str = "IterableDataset"


class FetcherFactory:
    """
    Factory for the fetcher.
    """

    @staticmethod
    def create_fetcher(dataset_type, dataset, auto_collation, collate_fn, drop_last=False):
        if dataset_type == DatasetType.MapDataset:
            return MapDatasetFetcher(dataset, auto_collation, collate_fn)
        if dataset_type == DatasetType.IterableDataset:
            return IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        raise ValueError(f"Unknown dataset type: {dataset_type}")


class DataLoader(Generic[_T_co]):
    """
    Data loader provides an iterator over the given dataset.

    It supports map style and iterable style dataset with single or multi-process loading.

    Args:
        dataset (Dataset): The dataset to load data from.
        batch_size (Union[int, None], optional): The number of samples per mini-batch.
            If ``None`` , will not batch. Default: ``1`` .
        shuffle (Union[bool, None], optional): Whether to shuffle the dataset. Default: ``None`` , not shuffle.
        sampler (Union[Sampler, Iterable, None], optional): The sampler to use. Default: ``None`` , use
            :class:`~mindspore.dataset.dataloader.SequentialSampler` if `shuffle` is ``False`` , or use
            :class:`~mindspore.dataset.dataloader.RandomSampler` .
        batch_sampler (Union[Sampler[List], Iterable[List], None], optional): The batch sampler to use.
            Default: ``None`` ,generate internal :class:`~mindspore.dataset.dataloader.BatchSampler` if `batch_size`
            is not ``None`` .
        num_workers (int, optional): The number of workers for loading. Default: ``0`` , load in main process.
        collate_fn (Union[_CollateFnType, None], optional): The collate function to use. Default: ``None`` , use
            default collate function.
        pin_memory (bool, optional): Whether to copy data into pinned memory. Default: ``False`` .
        drop_last (bool, optional): Whether to drop the last incomplete batch. Default: ``False`` .
        timeout (float, optional): The timeout for waiting the worker to process the data.
            Default: ``0.`` , wait forever.
        worker_init_fn (Union[Callable[[int], None], None], optional): The worker init function to use.
            Default: ``None`` , do nothing.
        multiprocessing_context (Union[multiprocessing.context.BaseContext, str, None], optional): The multiprocessing
            context to use. Default: ``None`` , use :mod:`mindspore.multiprocessing` .
        generator (Union[numpy.random.Generator, None], optional): The generator to use. Default: ``None`` ,
            use default generator.

    Keyword Args:
        prefetch_factor (Union[int, None], optional): The prefetch factor.
            Default: ``None`` , use ``2`` when `num_workers` is greater than ``0`` .
        persistent_workers (bool, optional): Whether to keep the worker alive after iteration. Default: ``False`` .
        in_order (bool, optional): Whether to keep the order of the data in multi-process loading. Default: ``True`` .

    Examples:
        >>> from mindspore.dataset.dataloader import DataLoader, Dataset, IterableDataset
        >>>
        >>> # 1. Load from map style dataset
        >>> class MapStyleDataset(Dataset):
        ...     def __init__(self, data):
        ...         self.data = data
        ...
        ...     def __getitem__(self, index):
        ...         return self.data[index]
        ...
        ...     def __len__(self):
        ...         return len(self.data)
        >>>
        >>> dataset = MapStyleDataset(range(2))
        >>> dataloader = DataLoader(dataset)
        >>> print(list(dataloader))
        [Tensor(shape=[1], dtype=Int64, value= [0]), Tensor(shape=[1], dtype=Int64, value= [1])]
        >>>
        >>> # 2. Load from iterable style dataset
        >>> class IterableStyleDataset(IterableDataset):
        ...     def __init__(self, num_samples):
        ...         self.start = 0
        ...         self.end = num_samples
        ...
        ...     def __iter__(self):
        ...         return iter(range(self.start, self.end))
        >>>
        >>> dataset = IterableStyleDataset(2)
        >>> dataloader = DataLoader(dataset)
        >>> print(list(dataloader))
        [Tensor(shape=[1], dtype=Int64, value= [0]), Tensor(shape=[1], dtype=Int64, value= [1])]
    """

    def __init__(
            self,
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
            generator: Union[np.random.Generator, None] = None,
            *,
            prefetch_factor: Union[int, None] = None,
            persistent_workers: bool = False,
            in_order: bool = True,
    ) -> None:

        if isinstance(dataset, IterableDataset):
            self.dataset_type = DatasetType.IterableDataset
        elif isinstance(dataset, Dataset):
            self.dataset_type = DatasetType.MapDataset
        else:
            if not hasattr(dataset, "__getitem__"):
                raise NotImplementedError(f"{dataset.__class__.__name__} should implement `__getitem__` method "
                                          f"if it is map style.")
            self.dataset_type = DatasetType.MapDataset

        if shuffle is not None and not isinstance(shuffle, bool):
            raise TypeError(f"shuffle must be bool, but got: {type(shuffle).__name__}")

        if not isinstance(pin_memory, bool):
            raise TypeError(f"pin_memory must be bool, but got: {type(pin_memory).__name__}")

        if not isinstance(num_workers, int) or isinstance(num_workers, bool):
            raise TypeError(f"num_workers must be int, but got: {type(num_workers).__name__}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, but got: {num_workers}")

        if not isinstance(timeout, int) and not isinstance(timeout, float) or isinstance(timeout, bool):
            raise TypeError(f"timeout must be int or float, but got: {type(timeout).__name__}")

        if timeout < 0:
            raise ValueError(f"timeout must be non-negative, but got: {timeout}")

        if num_workers == 0 and prefetch_factor is not None:
            raise ValueError("prefetch_factor must only be set when num_workers is greater than 0.")
        if num_workers > 0 and prefetch_factor is None:
            prefetch_factor = 2
        elif prefetch_factor is not None and prefetch_factor < 0:
            raise ValueError("prefetch_factor must be non-negative.")

        if multiprocessing_context is not None:
            if num_workers <= 0:
                raise ValueError(f"multiprocessing_context can only be specified when num_workers is greater than 0, "
                                 f"but got: {num_workers}.")
            if isinstance(multiprocessing_context, str):
                self.multiprocessing_context = ms.multiprocessing.get_context(multiprocessing_context)
            elif isinstance(multiprocessing_context, multiprocessing.context.BaseContext):
                self.multiprocessing_context = multiprocessing_context
            else:
                raise TypeError(f"multiprocessing_context must be a valid start method or "
                                f"multiprocessing.context.BaseContext, but got: {multiprocessing_context}.")
        else:
            self.multiprocessing_context = ms.multiprocessing

        if generator is not None and not isinstance(generator, np.random.Generator):
            raise TypeError(f"generator must be numpy.random.Generator, but got: {type(generator).__name__}")

        if prefetch_factor is not None:
            if not isinstance(prefetch_factor, int) or isinstance(prefetch_factor, bool):
                raise TypeError(f"prefetch_factor must be int, but got: {type(prefetch_factor).__name__}")
            if prefetch_factor <= 0:
                raise ValueError(f"prefetch_factor must be positive, but got: {prefetch_factor}")

        if not isinstance(persistent_workers, bool):
            raise TypeError(f"persistent_workers must be bool, but got: {type(persistent_workers).__name__}")
        if persistent_workers and num_workers <= 0:
            raise ValueError(f"persistent_workers can only be specified when num_workers is greater than 0, "
                             f"but got: {num_workers}")

        if not isinstance(in_order, bool):
            raise TypeError(f"in_order must be bool, but got: {type(in_order).__name__}")

        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.generator = generator if generator is not None else np.random.default_rng()
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.timeout = timeout
        self.in_order = in_order

        if self.dataset_type == DatasetType.IterableDataset:
            if sampler is not None:
                raise ValueError(f"DataLoader with IterableDataset: expected unspecified sampler option, "
                                 f"but got sampler={sampler}")
            if batch_sampler is not None:
                raise ValueError(f"DataLoader with IterableDataset: expected unspecified batch_sampler option, "
                                 f"but got batch_sampler={batch_sampler}")
            if shuffle not in {False, None}:
                raise ValueError(f"DataLoader with IterableDataset: expected unspecified shuffle option, "
                                 f"but got shuffle={shuffle}")

        if shuffle and sampler is not None:
            raise ValueError("`shuffle` and `sampler` can not specify at the same time.")

        if batch_sampler is not None and (batch_size != 1 or drop_last or sampler is not None or shuffle):
            raise ValueError("`batch_sampler` can not specify with `batch_size`, `drop_last`, `shuffle` or `sampler`.")

        if sampler is None:
            if self.dataset_type == DatasetType.IterableDataset:
                sampler = InfiniteSampler()
            else:
                if not shuffle:
                    sampler = SequentialSampler(self.dataset)
                else:
                    sampler = RandomSampler(self.dataset, generator=self.generator)

        self.sampler = sampler

        if batch_sampler is not None:
            self.batch_sampler = batch_sampler
            self.auto_collation = True
        elif batch_size is not None:
            self.batch_sampler = BatchSampler(self.sampler, batch_size, self.drop_last)
            self.auto_collation = True
        else:
            self.auto_collation = False

        if self.auto_collation:
            self.index_sampler = self.batch_sampler
        else:
            self.index_sampler = self.sampler

    def __iter__(self):
        if self.num_workers > 0:
            if self.persistent_workers:
                if not hasattr(self, "_iterator"):
                    self._iterator = _MultiProcessIterator(self)
                else:
                    self._iterator.reset()
                return self._iterator
            return _MultiProcessIterator(self)
        return _SingleProcessIterator(self)

    def __len__(self):
        if self.dataset_type == DatasetType.IterableDataset:
            if self.batch_size is not None:
                if self.drop_last:
                    return len(self.dataset) // self.batch_size
                return (len(self.dataset) - 1) // self.batch_size + 1
            return len(self.dataset)
        return len(self.index_sampler)


class _Iterator(Generic[_T_co]):
    """
    Iterator for the data loader.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        self.dataset = dataloader.dataset
        self.drop_last = dataloader.drop_last
        self.dataset_type = dataloader.dataset_type
        self.auto_collation = dataloader.auto_collation
        self.index_sampler = dataloader.index_sampler
        self.sampler_iterator = iter(self.index_sampler)
        self.collate_fn = dataloader.collate_fn
        if dataloader.pin_memory and ms.get_current_device().device_target == "CPU":
            logger.warning("Set pin_memory as True on CPU device will not have any effect.")
            self.pin_memory = False
        else:
            self.pin_memory = dataloader.pin_memory
        self._data_count = 0
        self.dataset_fetcher = FetcherFactory.create_fetcher(self.dataset_type, self.dataset, self.auto_collation,
                                                             self.collate_fn, self.drop_last)

    def __iter__(self):
        return self

    def __len__(self) -> int:
        return len(self.index_sampler)

    def __next__(self) -> Any:
        return self._get_next_data()

    def _get_next_data(self):
        """Get the next data."""
        raise NotImplementedError(f"{self.__class__.__name__} should implement `_get_next_data` method.")

    def _get_next_index(self):
        """Get the next index."""
        return next(self.sampler_iterator)

    def reset(self):
        """Reset the data loader."""
        self.sampler_iterator = iter(self.index_sampler)
        self._data_count = 0


class _SingleProcessIterator(_Iterator):
    """
    Iterator for the data loader in single process.
    """

    def _get_next_data(self):
        """Get the next data."""
        next_index = self._get_next_index()
        data = self.dataset_fetcher.fetch(next_index)
        if self.pin_memory:
            data = pin_memory_fn(data)
        return data


class _MultiProcessIterator(_Iterator):
    """
    Iterator for the data loader in multi process.
    """

    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__(dataloader)
        self.num_workers = dataloader.num_workers
        self._check_num_workers()
        self.multiprocessing_context = dataloader.multiprocessing_context
        self.worker_init_fn = dataloader.worker_init_fn
        self.persistent_workers = dataloader.persistent_workers
        self.prefetch_factor = dataloader.prefetch_factor
        self.timeout = dataloader.timeout
        self.in_order = dataloader.in_order
        self._base_seed = int(dataloader.generator.integers(low=0, high=np.iinfo(np.int64).max + 1, dtype=np.int64))

        self._setup_multiprocessing()

    def _check_num_workers(self):
        """Check the number of workers."""
        if hasattr(os, "sched_getaffinity"):
            get_affinity = True
            max_num_workers = len(os.sched_getaffinity(0))
        else:
            get_affinity = False
            max_num_workers = os.cpu_count()

        if self.num_workers > max_num_workers:
            if get_affinity:
                cpu_info = f"CPUs {max_num_workers} in the CPU set the current process is restricted to"
            else:
                cpu_info = f"logical CPUs {max_num_workers} in the system"
            logger.warning(
                f"DataLoader's `num_workers` with value {self.num_workers} is set too high, exceeding the number of "
                f"{cpu_info}, which may lead to competition for resources and slow down performance of DataLoader. "
                f"It is recommended to reduce the value of `num_workers`.")

    def _setup_multiprocessing(self):
        """Setup the multiprocessing."""
        self.data_workers = []
        self.index_queues = []
        self.data_queue = self.multiprocessing_context.Queue()
        self.worker_done = self.multiprocessing_context.Event()
        self.is_terminated = False
        if (self.multiprocessing_context != ms.multiprocessing
                and self.multiprocessing_context.get_start_method() == "fork"):
            logger.warning("multiprocessing_context does not currently support Python native and custom ForkContext, "
                           "switch to mindspore.multiprocessing instead.")
            self.multiprocessing_context = ms.multiprocessing
            self.multiprocessing_context.set_start_method("fork", force=True)
        for worker_id in range(self.num_workers):
            index_queue = self.multiprocessing_context.Queue()
            data_worker = self.multiprocessing_context.Process(
                target=data_worker_fn,
                args=(
                    self.dataset,
                    self.dataset_fetcher,
                    self.num_workers,
                    worker_id,
                    index_queue,
                    self.data_queue,
                    self.worker_done,
                    self.worker_init_fn,
                    self._base_seed,
                    self.persistent_workers,
                ),
                name=f"DataLoaderWorker{worker_id}",
                daemon=True,
            )
            data_worker.start()
            self.data_workers.append(data_worker)
            self.index_queues.append(index_queue)

        if self.pin_memory:
            self.result_queue = queue.Queue()
            device_id = ms.get_current_device().device_id
            self.pin_memory_done = threading.Event()
            self.pin_worker_thread = threading.Thread(
                target=pin_worker_fn,
                args=(
                    self.data_queue,
                    self.result_queue,
                    device_id,
                    self.pin_memory_done,
                ),
                name="PinMemoryWorker",
                daemon=True,
            )
            self.pin_worker_thread.start()
        else:
            self.result_queue = self.data_queue

        cde.register_worker_pids(id(self), [os.getpid()] + [data_worker.pid for data_worker in self.data_workers])
        set_sigchld_handler()

        self.reset()

    def reset(self):
        """Reset the data loader."""
        super().reset()
        resume_iteration = hasattr(self, "last_worker_assigned") and self.last_worker_assigned != -1
        self.order_index = 0
        self.next_data_index = 0
        self.task_info = {}
        self.last_worker_assigned = -1
        self.task_to_be_done = [0 for _ in range(self.num_workers)]
        self.worker_status = [True for _ in range(self.num_workers)]
        if resume_iteration:
            for index_queue in self.index_queues:
                index_queue.put(ResumeIterationFlag())
            handshake_count = len(self.index_queues)
            while handshake_count > 0:
                return_idx, _ = self._get_data_from_queue()
                if isinstance(return_idx, ResumeIterationFlag):
                    handshake_count -= 1
        for _ in range(self.num_workers * self.prefetch_factor):
            self._try_assign_one_task()

    def _try_assign_one_task(self):
        """
        Try to assign one task.
        """
        try:
            data_index = self._get_next_index()
        except StopIteration:
            return

        # find the next valid worker
        for i in range(self.num_workers):
            worker_to_assigned = (self.last_worker_assigned + i + 1) % self.num_workers
            if self.worker_status[worker_to_assigned]:
                if (self.in_order or self.task_to_be_done[worker_to_assigned]
                        < sum(self.task_to_be_done) // sum(self.worker_status) + 1):
                    self.index_queues[worker_to_assigned].put((self.order_index, data_index))
                    self.task_info[self.order_index] = (worker_to_assigned,)
                    self.order_index += 1
                    self.task_to_be_done[worker_to_assigned] += 1
                    self.last_worker_assigned = worker_to_assigned
                    return

    def _get_next_data(self):
        """
        Get the next data.
        """
        # pylint: disable=too-many-nested-blocks
        while self.next_data_index < self.order_index:
            if self.next_data_index not in self.task_info:
                self.next_data_index += 1
                continue
            else:
                task = self.task_info.get(self.next_data_index)
                if len(task) == 2:  # task finished
                    worker_id, data = task
                    self.task_info.pop(self.next_data_index)
                    self.next_data_index += 1
                    self.task_to_be_done[worker_id] -= 1
                    self._try_assign_one_task()
                    if isinstance(data, WorkerException):
                        data.reraise()
                    return data

                worker_id = task[0]
                if not self.worker_status[worker_id]:  # invalid index
                    self.task_info.pop(self.next_data_index)
                    self.next_data_index += 1
                    continue
                else:  # worker not finished
                    while True:
                        order_index, data = self._get_data_from_queue()
                        if self.dataset_type == DatasetType.IterableDataset and data is None:
                            worker_id = self.task_info.pop(order_index)[0]
                            if not self.persistent_workers:
                                self.index_queues[worker_id].put(None)
                            self.worker_status[worker_id] = False
                            self._try_assign_one_task()
                            break
                        if order_index != self.next_data_index:  # not we want
                            if not self.in_order:
                                worker_id = self.task_info.pop(order_index)[0]
                                self.task_to_be_done[worker_id] -= 1
                                self._try_assign_one_task()
                                if isinstance(data, WorkerException):
                                    data.reraise()
                                return data

                            if isinstance(data, WorkerException):
                                data.reraise()
                            self.task_info[order_index] += (data,)
                        else:
                            worker_id = self.task_info.pop(self.next_data_index)[0]
                            self.next_data_index += 1
                            self.task_to_be_done[worker_id] -= 1
                            self._try_assign_one_task()
                            if isinstance(data, WorkerException):
                                data.reraise()
                            return data
        if not self.persistent_workers:
            self.terminate()
        raise StopIteration

    def _get_data_from_queue(self):
        """
        Get the data from the queue.
        """
        if self.timeout > 0:
            status, data = self._try_get_data(self.timeout)
            if status:
                return data
            raise RuntimeError(f"DataLoader get data timeout after {self.timeout} seconds.")

        while True:
            if self.pin_memory and not self.pin_worker_thread.is_alive():
                raise RuntimeError("DataLoader pin memory thread is not alive.")
            success, data = self._try_get_data()
            if success:
                return data

    def _try_get_data(self, timeout=WORKER_TIME_OUT):
        """
        Try to get the data from the queue.
        """
        try:
            data = self.result_queue.get(timeout=timeout)
            return True, data
        except Exception as exc:  # pylint: disable=W0703
            failed_workers = []
            for worker_id, w in enumerate(self.data_workers):
                if self.worker_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self.index_queues[worker_id].put(None)
                    self.worker_status[worker_id] = False
            if failed_workers:
                pids_str = ", ".join(str(w.pid) for w in failed_workers)
                raise RuntimeError(f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly") from exc
            if isinstance(exc, queue.Empty):
                return False, None

    def terminate(self):
        """
        Terminate the data loader.
        """
        if not self.is_terminated:
            self.is_terminated = True
            try:
                if self.pin_memory:
                    self.pin_memory_done.set()
                    self.data_queue.put((None, None))
                    self.pin_worker_thread.join()
                    self.data_queue.close()
                    self.data_queue.join_thread()

                self.worker_done.set()
                for worker_id in range(len(self.data_workers)):
                    if self.persistent_workers or self.worker_status[worker_id]:
                        self.index_queues[worker_id].put(None)
                        self.worker_status[worker_id] = False
                for worker in self.data_workers:
                    worker.join(timeout=WORKER_TIME_OUT)
                for index_queue in self.index_queues:
                    index_queue.close()
                    index_queue.join_thread()
            except Exception as e:
                raise e
            finally:
                cde.deregister_worker_pids(id(self))
                for worker in self.data_workers:
                    if worker.is_alive():
                        logger.warning(f"terminate worker {worker.pid}")
                        worker.terminate()

    def __del__(self):
        self.terminate()
