# Copyright 2024 Huawei Technologies Co., Ltd
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
import stat

import pytest
import os
import mindspore as ms
import mindspore.nn as nn

from mindspore.common.file_system import FileSystem
from tests.mark_utils import arg_mark


class Network(nn.Cell):
    def __init__(self, lin_weight, lin_bias):
        super().__init__()
        self.lin = nn.Dense(2, 3, weight_init=lin_weight, bias_init=lin_bias)
        self.relu = nn.ReLU()

    def construct(self, x):
        out = self.lin(x)
        out = self.relu(out)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ckpt_save_with_crc(mode):
    """
    Feature: Save ckpt with crc check.
    Description: Save ckpt with crc check.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    weight = ms.Tensor([[0.27201429, 2.22499485],
                        [-0.5636731, -2.21354142],
                        [1.3987198, 0.04099071]], dtype=ms.float32)
    bias = ms.Tensor([-0.41271235, 0.28378568, -0.81612898], dtype=ms.float32)
    net = Network(weight, bias)
    ms.save_checkpoint(net, f"./save_with_crc_{mode}.ckpt", crc_check=True)
    ms.load_checkpoint(f"./save_with_crc_{mode}.ckpt", crc_check=True)
    ms.load_checkpoint(f"./save_with_crc_{mode}.ckpt", crc_check=False)
    os.chmod(f"./save_with_crc_{mode}.ckpt", stat.S_IWRITE)
    os.remove(f"./save_with_crc_{mode}.ckpt")


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ckpt_save_with_crc_failed(mode):
    """
    Feature: Save ckpt with crc check.
    Description: Save ckpt with crc check.
    Expectation: Success.
    """
    ms.set_context(mode=mode)
    weight = ms.Tensor([[0.27201429, 2.22499485], [-0.5636731, -2.21354142], [1.3987198, 0.04099071]], dtype=ms.float32)
    bias = ms.Tensor([-0.41271235, 0.28378568, -0.81612898], dtype=ms.float32)
    net = Network(weight, bias)
    ms.save_checkpoint(net, f'./save_with_crc_failed_{mode}.ckpt', crc_check=True)

    _ckpt_fs = FileSystem()
    os.chmod(f"./save_with_crc_failed_{mode}.ckpt", stat.S_IWRITE)
    with _ckpt_fs.open(f"./save_with_crc_failed_{mode}.ckpt", *_ckpt_fs.create_args) as f:
        f.write(b"111")

    with pytest.raises(ValueError):
        ms.load_checkpoint(f"./save_with_crc_failed_{mode}.ckpt", crc_check=True)
    os.chmod(f'./save_with_crc_failed_{mode}.ckpt', stat.S_IWRITE)
    os.remove(f'./save_with_crc_failed_{mode}.ckpt')
