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
Worker module.
"""

import os
import random
import sys
import traceback
from queue import Empty

import numpy as np

import mindspore
import mindspore._c_dataengine as cde
from mindspore import log as logger
from . import WORKER_TIME_OUT

worker_info_local = None


class ResumeIterationFlag:
    """
    Flag for resume iteration.
    """


# The function `_generate_state` is adapted from `numpy.random.SeedSequence`
# from https://github.com/numpy/numpy/blob/main/numpy/random/bit_generator.pyx
# It's MIT licensed, here is the copyright:

# Copyright (c) 2015 Melissa E. O'Neill
# Copyright (c) 2019 NumPy Developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# This function generates an array of int32 as the seed for
# `numpy.random`, in order to prevent state collision due to same
# seed and algorithm for `numpy.random` and `random` modules.
def _generate_state(base_seed, worker_id):
    """
    Generate the state for the random number generator.
    """
    INIT_A = 0x43B0D7E5
    MULT_A = 0x931E8875
    INIT_B = 0x8B51F9DD
    MULT_B = 0x58F38DED
    MIX_MULT_L = 0xCA01F9DD
    MIX_MULT_R = 0x4973F715
    XSHIFT = np.dtype(np.uint32).itemsize * 8 // 2
    MASK32 = 0xFFFFFFFF

    entropy = [worker_id, base_seed & MASK32, base_seed >> 32, 0]
    pool = [0] * 4

    hash_const_A = INIT_A

    def hash(value):
        nonlocal hash_const_A
        value = (value ^ hash_const_A) & MASK32
        hash_const_A = (hash_const_A * MULT_A) & MASK32
        value = (value * hash_const_A) & MASK32
        value = (value ^ (value >> XSHIFT)) & MASK32
        return value

    def mix(x, y):
        result_x = (MIX_MULT_L * x) & MASK32
        result_y = (MIX_MULT_R * y) & MASK32
        result = (result_x - result_y) & MASK32
        result = (result ^ (result >> XSHIFT)) & MASK32
        return result

    # Add in the entropy to the pool.
    for i in range(len(pool)):
        pool[i] = hash(entropy[i])

    # Mix all bits together so late bits can affect earlier bits.
    for i_src in range(len(pool)):
        for i_dst in range(len(pool)):
            if i_src != i_dst:
                pool[i_dst] = mix(pool[i_dst], hash(pool[i_src]))

    hash_const_B = INIT_B
    state = []
    for i_dst in range(4):
        data_val = pool[i_dst]
        data_val = (data_val ^ hash_const_B) & MASK32
        hash_const_B = (hash_const_B * MULT_B) & MASK32
        data_val = (data_val * hash_const_B) & MASK32
        data_val = (data_val ^ (data_val >> XSHIFT)) & MASK32
        state.append(data_val)
    return state


class WorkerInfo:
    """
    Worker information.
    """

    _initialized = False

    def __init__(self, id, num_workers, seed, dataset):
        self.id = id
        self.num_workers = num_workers
        self.seed = seed
        self.dataset = dataset
        self._initialized = True

    def __setattr__(self, key, value):
        if self._initialized:
            raise RuntimeError("Cannot modify the attributes of WorkerInfo object after initialization.")
        return super().__setattr__(key, value)

    def __repr__(self):
        return (f"WorkerInfo: {{id: {self.id}, num_workers: {self.num_workers}, "
                f"seed: {self.seed}, dataset: {self.dataset}}}")


class KeyErrorMsg(str):
    """
    Key error message.
    """

    __slots__ = ()

    def __repr__(self):
        """
        Return the string representation of the exception.
        """

        return self


class WorkerException:
    """
    Worker exception.
    """

    def __init__(self, worker_id=None, exc_info=None):
        self.worker_id = worker_id
        self.pid = os.getpid()
        exc_info = exc_info if exc_info is not None else sys.exc_info()
        self.exc_type = exc_info[0]
        self.exc_msg = "".join(traceback.format_exception(*exc_info))

    def reraise(self):
        """
        Reraise the exception.
        """

        if self.worker_id is not None:
            process_msg = f"DataLoader worker {self.worker_id}"
        else:
            process_msg = "DataLoader main process"
        exc_msg = (process_msg + f" (pid: {self.pid}) caught {self.exc_type.__name__} with message:\n{self.exc_msg}")
        if self.exc_type == KeyError:
            exc_msg = KeyErrorMsg(exc_msg)
        try:
            raise self.exc_type(message=exc_msg)
        except Exception:
            raise RuntimeError(exc_msg) from None


class ParentProcessMonitor:
    """
    Parent process monitor.
    """

    def __init__(self):
        self.ppid = os.getppid()

    def is_alive(self):
        """
        Check if the parent process is alive.
        """

        return os.getppid() == self.ppid


def data_worker_fn(dataset, fetcher, num_workers, worker_id, index_queue, data_queue, worker_done, worker_init_fn,
                   base_seed, persistent_workers):
    """
    Data worker function.
    """

    try:
        try:
            cde.register_worker_handlers()
            mindspore.device_context.cpu.op_tuning.threads_num(1)

            worker_seed = base_seed + worker_id
            random.seed(worker_seed)
            mindspore.manual_seed(worker_seed)
            np.random.seed(_generate_state(base_seed, worker_id))

            global worker_info_local
            worker_info_local = WorkerInfo(id=worker_id, num_workers=num_workers, seed=worker_seed, dataset=dataset)

            if worker_init_fn is not None:
                worker_init_fn(worker_id)
            fetcher.reset()
        except Exception:  # pylint: disable=W0703
            exc = WorkerException(worker_id)
            data_queue.put((-1, exc))
            return

        iteration_finished = False
        parent_process_monitor = ParentProcessMonitor()

        while parent_process_monitor.is_alive():
            try:
                index_item = index_queue.get(timeout=WORKER_TIME_OUT)
            except Empty:
                continue
            if isinstance(index_item, ResumeIterationFlag):
                data_queue.put((index_item, None))
                iteration_finished = False
                fetcher.reset()
                continue
            elif index_item is None:
                if not worker_done.is_set():
                    raise RuntimeError("Got None from index.")
                break  # we got the last data of index queue, now can safely quit
            elif worker_done.is_set() or iteration_finished:
                # main process send quit flag, but we still need to empty the index queue, skip get data from dataset
                continue
            order_index, data_index = index_item
            try:
                data = fetcher.fetch(data_index)
            except StopIteration:
                iteration_finished = True
                data_queue.put((order_index, None))
                if not persistent_workers:
                    break
                else:
                    continue
            except Exception:  # pylint: disable=W0703
                data = WorkerException(worker_id)
            data_queue.put((order_index, data))
            del order_index, data_index, data, index_item

    except KeyboardInterrupt:
        logger.info(f"DataLoader worker {worker_id} (pid: {os.getpid()}) was interrupted by the keyboard.")
    if worker_done.is_set():
        data_queue.close()
        data_queue.join_thread()


def get_worker_info():
    """
    Get the information about the current DataLoader worker process.

    The information includes:

    - worker_id (int): The id of the current worker.
    - num_workers (int): The total number of workers.
    - seed (int): The random seed used by the current worker.
            This value is determined by the base seed generated by the main process and the worker id.
    - dataset (Dataset): The dataset object copied from the main process to the current worker.

    If the current process is not a DataLoader worker process, return ``None``.

    Returns:
        Union[WorkerInfo, None], the information about the current DataLoader worker process.
    """
    return worker_info_local
