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
import multiprocessing
import multiprocessing.context
import queue
import numbers
import os
import itertools
from typing import Any, AnyStr, Callable, Generic, Iterable, List, Mapping, Optional, overload, Protocol, \
    Sequence, TypeVar, Union

import numpy as np
import mindspore as ms
from mindspore import log as logger
from mindspore.common import Tensor
from mindspore.common.generator import Generator

from .dataset import Dataset, IterableDataset
from .sampler import BatchSampler, RandomSampler, Sampler, SequentialSampler, InfiniteSampler
from .utils.worker import worker_loop, _IterableDatasetStopIteration
from .utils.fetch import _MapDatasetFetcher, _IterableDatasetFetcher
from .utils.collate import default_collate, default_convert

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_K = TypeVar('_K')
_V = TypeVar('_V')


def _get_distributed_settings():
    if ms.mint.distributed.is_available() and ms.mint.distributed.is_initialized():
        return ms.mint.distributed.get_world_size(), ms.mint.distributed.get_rank()
    else:
        return 1, 0


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
    def create_fetcher(dataset_type, dataset, auto_collation, collate_fn, drop_last=False):
        if dataset_type == DatasetType.MapDataset:
            return _MapDatasetFetcher(dataset, auto_collation, collate_fn)
        elif dataset_type == DatasetType.IterableDataset:
            return _IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            raise ValueError("Unknown dataset type: {}".format(dataset_type))


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
                 persistent_workers: bool = False,
                 pin_memory_device: str = "",
                 ) -> None:
        self.dataset = dataset
        self.drop_last = drop_last
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.generator = generator
        self.collate_fn = collate_fn
        self.prefetch_factor = prefetch_factor
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.persistent_workers = persistent_workers

        if self.num_workers < 0:
            raise ValueError(
                f"`num_workers` should not be less than 0."
            )
        elif self.num_workers == 0 and self.prefetch_factor is not None:
            raise ValueError(
                f"`prefetch_factor` could only be specified in multiprocessing mode, set `num_workers` > 0 to enable multiprocessing."
            )
        elif self.num_workers > 0 and self.prefetch_factor is None:
            self.prefetch_factor = 2
        elif self.num_workers > 0 and self.prefetch_factor < 0:
            raise ValueError(
                f"`prefetch_factor` should not be less than 0."
            )
        

        if isinstance(dataset, IterableDataset):
            self.dataset_type = DatasetType.IterableDataset
        else:
            self.dataset_type = DatasetType.MapDataset
        
        if self.dataset_type == DatasetType.IterableDataset:
            if sampler is not None:
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified sampler option, but got sampler={sampler}"
                )
            if batch_sampler is not None:
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified batch_sampler option, but got batch_sampler={batch_sampler}"
                )
            if shuffle not in {False, None}:
                raise ValueError(
                    f"DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle={shuffle}"
                )
            sampler = InfiniteSampler()

        if sampler is not None and shuffle:
            raise ValueError("`shuffle` and `sampler` can not specify at the same time.")

        if batch_sampler is not None:
            if batch_size != 1 or shuffle or drop_last or sampler is not None:
                raise ValueError("`batch_sampler` can not specify with `batch_size`, `drop_last`, `shuffle` or `sampler`.")
        
        if batch_size is None and drop_last:
            raise ValueError("when `batch_size` is None, `drop_last` can not be specified")
        
        if sampler is None:
            if shuffle:
                sampler = RandomSampler(self.dataset, generator=self.generator)
            else:
                sampler = SequentialSampler(self.dataset)

        self.sampler = sampler

        if batch_sampler is not None:
            self.batch_sampler = batch_sampler
            self.auto_collation = True
        elif batch_size is not None:
            self.batch_sampler = BatchSampler(sampler, batch_size, self.drop_last)
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
                if hasattr(self, "_iterator"):
                    self._iterator._reset(self)
                else:
                    self._iterator = _MultiProcessIterator(self)
                return self._iterator
            else:
                return _MultiProcessIterator(self)
        else:
            return _SingleProcessIterator(self)

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if isinstance(multiprocessing_context, str):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            "multiprocessing_context option "
                            f"should specify a valid start method in {valid_start_methods!r}, but got "
                            f"multiprocessing_context={multiprocessing_context!r}"
                        )
                    multiprocessing_context = multiprocessing.get_context(
                        multiprocessing_context
                    )

                if not isinstance(
                        multiprocessing_context, multiprocessing.context.BaseContext
                ):
                    raise TypeError(
                        "multiprocessing_context option should be a valid context "
                        "object or a string specifying the start method, but got "
                        f"multiprocessing_context={multiprocessing_context}"
                    )
            else:
                raise ValueError(
                    "multiprocessing_context can only be used with "
                    "multi-process loading (num_workers > 0), but got "
                    f"num_workers={self.num_workers}"
                )

        self.__multiprocessing_context = multiprocessing_context


class _Iterator(Generic[_T_co]):
    def __init__(self, dataloader: DataLoader) -> None:
        self.dataset = dataloader.dataset
        self.drop_last = dataloader.drop_last
        self.dataset_type = dataloader.dataset_type
        self.num_workers = dataloader.num_workers
        self.auto_collation = dataloader.auto_collation
        self.index_sampler = dataloader.index_sampler
        self.sampler_iterator = iter(self.index_sampler)
        self.collate_fn = dataloader.collate_fn
        self.world_size, self.rank = _get_distributed_settings()
        #self.base_seed = (
        #    Tensor(1).random_(generator=dataloader.generator).item()
        #)
        self.base_seed = 0
        self.shared_seed = None
        self.pin_memory = dataloader.pin_memory
        self.timeout = dataloader.timeout
        self.persistent_workers = dataloader.persistent_workers


    def __iter__(self):
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
                                                             self.collate_fn, self.drop_last)

    def _get_next_data(self):
        next_index = self._get_next_index()
        return self.dataset_fetcher.fetch(next_index)


class _MultiProcessIterator(_Iterator):
    def __init__(self, dataloader: DataLoader) -> None:
        super().__init__(dataloader)
        self._check_num_workers()

        # self-defined multiprocessing
        if dataloader.multiprocessing_context is None:
            multiprocessing_context = multiprocessing
        else:
            multiprocessing_context = dataloader.multiprocessing_context

        self.prefetch_factor = dataloader.prefetch_factor
        self.worker_init_fn = dataloader.worker_init_fn

        # No certainty which module multiprocessing_context is
        self.worker_result_queue = multiprocessing_context.Queue()
        self.worker_pids_set = False
        self.shutdown = False
        self.workers_done_event = multiprocessing_context.Event()

        self.index_queues = []
        self.workers = []
        logger.warning(f"start to create subprocess, num_workers:{self.num_workers}, prefetch_factor:{self.prefetch_factor}")

        for i in range(self.num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=worker_loop,
                args=(
                    self.dataset_type,
                    self.dataset,
                    index_queue,
                    self.worker_result_queue,
                    self.workers_done_event,
                    self.auto_collation,
                    self.collate_fn,
                    self.drop_last,
                    self.base_seed,
                    self.worker_init_fn,
                    i,
                    self.num_workers,
                    self.persistent_workers,
                    self.shared_seed,
                ),
                name=f"DataWorker{i}"
            )
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self.index_queues.append(index_queue)
            self.workers.append(w)

        if self.pin_memory:
            raise RuntimeError("not support pin_memory")
        else:
            self.data_queue = self.worker_result_queue
        logger.warning(f"create subprocess success")
        self.reset()

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

    def reset(self):
        self.send_idx = 0  # idx of the next task to be sent to workers
        self.rcvd_idx = 0  # idx of the next task to be returned in __next__
        # information about data not yet yielded, i.e., tasks w/ indices in range [rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self.task_info = {}
        self.tasks_outstanding = (
            0  # always equal to count(v for v in task_info.values() if len(v) == 1)
        )
        # A list of booleans representing whether each worker still has work to
        # do, i.e., not having exhausted its iterable dataset object. It always
        # contains all `True`s if not using an iterable-style dataset
        # (i.e., if kind != Iterable).
        # Not that this indicates that a worker still has work to do *for this epoch*.
        # It does not mean that a worker is dead. In case of `_persistent_workers`,
        # the worker will be reset to available in the next epoch.
        self.workers_status = [True for i in range(self.num_workers)]
        # A list of integers representing how many tasks are outstanding for each worker
        # Incremented when a task is dispatched to the worker
        # Decremented when that data has been given to the main thread
        # Each worker should have at most self.prefetch_factor tasks outstanding
        self.workers_num_tasks = [0 for i in range(self.num_workers)]
        # Reset the worker queue cycle so it resumes next epoch at worker 0
        self.worker_queue_idx_cycle = itertools.cycle(range(self.num_workers))
        # prime the prefetch loop
        for i in range(self.prefetch_factor * self.num_workers):
            logger.warning(f"call _try_put_index put index to worker")
            self._try_put_index()

    def _try_put_index(self):
        max_tasks = self.prefetch_factor * self.num_workers
        assert self.tasks_outstanding < max_tasks

        # 如果是默认map sampler、默认batchsampler、自定义map/iter sampler、自定义batchsampler都可以保证StopIteration
        # 但如果是默认iter sampler将是个死循环，sample_idx永远是None
        try:
            sample_idx = next(self.sampler_iterator)
        except StopIteration:
            return

        # 在某些worker退出后，如果其他worker还在工作时，把index塞给他，否则会丢失这次的索引
        for _ in range(self.num_workers):
            # 轮询worker状态
            worker_queue_idx = next(self.worker_queue_idx_cycle)
            if self.workers_status[worker_queue_idx] is True:
                break

        logger.warning(f"In _try_put_index, sample_idx:{sample_idx}, worker_queue_idx:{worker_queue_idx}")
        #breakpoint()

        self.index_queues[worker_queue_idx].put((self.send_idx, sample_idx))  # type: ignore[possibly-undefined]
        self.task_info[self.send_idx] = (worker_queue_idx,)
        self.workers_num_tasks[worker_queue_idx] += 1
        self.tasks_outstanding += 1
        self.send_idx += 1
 

    def _try_get_data(self, timeout=5):
        try:
            logger.warning(f"In _try_get_data, before data_queue.get")
            data = self.data_queue.get(timeout=timeout)
            logger.warning(f"In _try_get_data, get data:{data}")
            return (True, data)
        except Exception as e:
            # 如果有子进程退出，则直接提示子进程异常退出
            failed_workers = []
            for worker_id, w in enumerate(self.workers):
                if self.workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
            if len(failed_workers) > 0:
                pids_str = ", ".join(str(w.pid) for w in failed_workers)
                raise RuntimeError(
                    f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
                ) from e
            # 正常超时empty
            if isinstance(e, queue.Empty):
                return (False, None)

    def _get_data(self):
        if self.timeout > 0:
            logger.warning(f"In _get_data, before _try_get_data")
            success, data = self._try_get_data(self.timeout)
            logger.warning(f"In _get_data, get {(success, data)}")
            if success:
                return data
            else:
                raise RuntimeError(
                    f"DataLoader timed out after {self.timeout} seconds"
                )
        else:
            while True:
                success, data = self._try_get_data()
                if success:
                    return data

    def _get_next_data(self):
        '''主进程从res_queue接收数据/异常'''
        while True:
            # 主进程取完最后一条数据，rcvd_idx应该等于send_idx，退出
            logger.warning(f"self.rcvd_idx: {self.rcvd_idx}, self.send_idx:{self.send_idx}")
            logger.warning(f"self.task_info: {self.task_info}")
            
            # 在Iterdataset的默认sampler InfiniteSampler结束时，主进程会发送一个None给
            while self.rcvd_idx < self.send_idx:
                info = self.task_info.get(self.rcvd_idx, None)
                if info:
                    worker_id = info[0]
                    if len(info) == 2 or self.workers_status[worker_id]:  # has data or is still active
                        break
                    logger.warning(f"remove {self.task_info[self.rcvd_idx]} from worker {worker_id}")
                    del self.task_info[self.rcvd_idx]
                self.rcvd_idx += 1
            else:
                # no valid `self.rcvd_idx` is found (i.e., didn't break)
                raise StopIteration

            # 如果在之前的_get_data提前取出来了未来需要的数据，则直接返回
            # 如果没之前存好，则正常get_data取
            if len(self.task_info[self.rcvd_idx]) == 2:
                logger.warning(f"when self.task_info store future data")
                worker_id, data = self.task_info.pop(self.rcvd_idx)
                self.rcvd_idx += 1
                return self._process_data(data, worker_id)
            else:
                idx, data = self._get_data()
            
            # 如果默认iter sampler在子进程中抛出了异常
            if isinstance(data, _IterableDatasetStopIteration):
                if self.dataset_type == DatasetType.IterableDataset:
                    #breakpoint()
                    logger.warning(f"In _get_next_data, worker_id {data.worker_id} raise StopIteration")
                    # 发送None到子进程，要求子进程退出
                    self._mark_worker_as_unavailable(data.worker_id)
                    # 如果这个worker提前stop了，其他worker可以继续处理index
                    self._process_data(None, data.worker_id)
                    continue

            logger.warning(f"In _get_next_data")
            # rcvd_idx代表下一条要获取的样本id，如果idx就是，则取出来
            if idx == self.rcvd_idx:
                worker_id = self.task_info.pop(idx)[0]
                self.rcvd_idx += 1
                return self._process_data(data, worker_id)
            # 如果不是下一条要的id但已经准备好了，先存起来
            else:
                self.task_info[idx] += (data,)

    def _process_data(self, data, worker_idx):
        #breakpoint()
        logger.warning(f"In _process_data, put next index to worker")
        self.workers_num_tasks[worker_idx] -= 1
        self.tasks_outstanding -= 1
        self._try_put_index()
        # 如果收到子进程的异常直接raise
        if isinstance(data, Exception):
            raise data
        return data

    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        assert self.workers_status[worker_id] or (
            self.persistent_workers and shutdown
        )

        # Signal termination to that specific worker.
        q = self.index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self.workers_status[worker_id] = False

        assert self.workers_done_event.is_set() == shutdown
