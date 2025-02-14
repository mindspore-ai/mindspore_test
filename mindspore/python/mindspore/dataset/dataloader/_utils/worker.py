import os
import sys
import random
import numpy as np
import queue
from dataclasses import dataclass
from typing import Optional

import mindspore as ms
from mindspore import log as logger

# mydebug
logger.warning = lambda *args, **kwargs: None


if sys.platform == "win32":
    raise RuntimeError("multiprocessing not support windows")

else:
    class ManagerWatchdog:  # type: ignore[no-redef]
        def __init__(self) -> None:
            self.manager_pid = os.getppid()
            self.manager_dead = False

        def is_alive(self):
            if not self.manager_dead:
                self.manager_dead = os.getppid() != self.manager_pid
            return not self.manager_dead


worker_info: Optional["WorkerInfo"] = None


class WorkerInfo:
    id: int
    num_workers: int
    seed: int
    dataset: "Dataset"
    __initialized = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__keys = tuple(kwargs.keys())
        self.__initialized = True

    def __setattr__(self, key, val):
        if self.__initialized:
            raise RuntimeError(
                f"Cannot assign attributes to {self.__class__.__name__} objects"
            )
        return super().__setattr__(key, val)

    def __repr__(self):
        items = [f"{k}={getattr(self, k)}" for k in self.__keys]
        return f"{self.__class__.__name__}({', '.join(items)})"


def get_worker_info() -> Optional[WorkerInfo]:
    r"""Returns the information about the current
    """
    return _worker_info


@dataclass(frozen=True)
class _IterableDatasetStopIteration:
    r"""Dummy class used to signal the end of an IterableDataset"""
    worker_id: int


def worker_loop(
    dataset_type,
    dataset,
    index_queue,
    data_queue,
    done_event,
    auto_collation,
    collate_fn,
    drop_last,
    base_seed,
    init_fn,
    worker_id,
    num_workers,
    persistent_workers,
    shared_seed,
):
    # 解决循环依赖
    from ..dataloader import FetcherFactory, DatasetType

    # See NOTE [ Data Loader Multiprocessing Shutdown Logic ] for details on the
    # logic of this function.
    logger.warning(f"come into subprocess, pid:{os.getpid()}")

    try:
        seed = base_seed + worker_id
        random.seed(seed)
        ms.set_seed(seed)
        np.random.seed(seed)

        global _worker_info
        _worker_info = WorkerInfo(
            id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset
        )

        init_exception = None

        try:
            if init_fn is not None:
                init_fn(worker_id)

            fetcher = FetcherFactory.create_fetcher(dataset_type, dataset, auto_collation,
                                                    collate_fn, drop_last)
        except Exception as e:
            init_exception = RuntimeError(f"Failed to init DataLoader DataWorker{worker_id}\nError msg: {str(e)}")
        logger.warning(f"craete a new fetcher:{fetcher}")

        watchdog = ManagerWatchdog()
        iteration_end = False

        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=5)
            except queue.Empty:
                logger.warning(f"this worker get data timeout")
                continue
            if r is None:
                # Received the final signal
                # 一旦fetcher.fetch抛出stop iteration，iteration_end会置为True
                assert done_event.is_set() or iteration_end
                logger.warning(f"this worker get None from  mainprocess, thus exit")
                break
            elif done_event.is_set() or iteration_end:
                # `done_event` is set. But I haven't received the final signal
                # (None) yet. I will keep continuing until get it, and skip the
                # processing steps.
                continue
            idx, sample_index = r
            logger.warning(f"get data from index_queue:{r}")

            if init_exception is not None:
                data = init_exception
                init_exception = None
            else:
                try:
                    data = fetcher.fetch(sample_index)  # type: ignore[possibly-undefined]
                except Exception as e:
                    # iterdataset，如果是默认sampler，只能在这里抛出异常
                    if (
                        isinstance(e, StopIteration)
                        and dataset_type == DatasetType.IterableDataset
                    ):
                        data = _IterableDatasetStopIteration(worker_id)
                        # Set `iteration_end`
                        #   (1) to save future `next(...)` calls, and
                        #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                        iteration_end = True
                    else:
                        data = RuntimeError(f"Failed in DataLoader worker process {worker_id}\nError msg: {str(e)}")
            data_queue.put((idx, data))
            logger.warning(f"put sample into data_queue: {(idx, data)}")
            del data, idx, sample_index, r  # save memory
    except KeyboardInterrupt:
        # Main process will raise KeyboardInterrupt anyways.
        pass
    if done_event.is_set():
        data_queue.cancel_join_thread()
        data_queue.close()
