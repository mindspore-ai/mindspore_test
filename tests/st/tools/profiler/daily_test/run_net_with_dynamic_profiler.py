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
# ============================================================================
"""build net"""

import os
import sys
import contextlib
import json
import argparse
import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore import nn
from mindspore.profiler.dynamic_profiler import DynamicProfilerMonitor


class Net(nn.Cell):
    """net"""

    def __init__(self):
        super().__init__()
        self.fc = nn.Dense(2, 2)

    def construct(self, x):
        """construct"""
        return self.fc(x)


def generator_net():
    """generator net"""
    for _ in range(20):
        yield np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32)


def train(net):
    """train"""
    optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
    data = ds.GeneratorDataset(generator_net(), ["data", "label"])
    model = ms.train.Model(net, loss, optimizer)
    model.train(1, data)


def _cleanup_dynamic_profiler(dp):
    """Force cleanup to avoid stale shared memory without modifying framework code."""
    # Best-effort framework cleanup with specific exceptions
    with contextlib.suppress(RuntimeError, OSError, ValueError):
        dp.on_train_end()

    # Stop background worker loop and join process
    loop_flag = getattr(dp, "_shared_loop_flag", None)
    if loop_flag is not None:
        loop_flag.value = False

    proc = getattr(dp, "_process", None)
    if proc is not None:
        with contextlib.suppress(RuntimeError, AssertionError):
            proc.join(timeout=2.0)

    # Unconditionally close and unlink shared memory / mmap with specific exceptions
    shm = getattr(dp, "_shm", None)
    if shm is not None:
        if sys.version_info >= (3, 8):
            with contextlib.suppress(OSError, BufferError):
                shm.close()
            with contextlib.suppress(OSError, FileNotFoundError, PermissionError):
                shm.unlink()
        else:
            with contextlib.suppress(OSError, ValueError):
                shm.close()
            mmf = getattr(dp, "_memory_mapped_file", None)
            if mmf is not None:
                with contextlib.suppress(OSError, ValueError):
                    mmf.close()
            fd = getattr(dp, "fd", None)
            if fd is not None:
                with contextlib.suppress(OSError):
                    os.close(fd)
            shm_path = getattr(dp, "_shm_path", None)
            if shm_path and os.path.exists(shm_path):
                with contextlib.suppress(OSError):
                    os.remove(shm_path)


def train_net_with_dynamic_profiler(output_path, cfg_path):
    """train net"""
    net = Net()
    step_num = 15
    dp = DynamicProfilerMonitor(cfg_path=cfg_path, output_path=output_path)
    try:
        for i in range(step_num):
            train(net)
            if i == 5:
                change_cfg_json(os.path.join(cfg_path, "profiler_config.json"))
            dp.step()
    finally:
        _cleanup_dynamic_profiler(dp)


def change_cfg_json(json_path):
    """change json"""
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data['start_step'] = 6
    data['stop_step'] = 7

    with open(json_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run net with dynamic profiler.')
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--cfg_path', type=str)
    args = parser.parse_args()
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")
    train_net_with_dynamic_profiler(output_path=args.output_path, cfg_path=args.cfg_path)
