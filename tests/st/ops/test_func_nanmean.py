# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops


class Net(nn.Cell):
    def construct(self, x):
        return ops.nanmean(x, axis=0, keepdims=True)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_nanmean(mode):
    """
    Feature: ops.nanmean
    Description: Verify the result of nanmean
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[float("nan"), 128.1, -256.9], [float("nan"), float("nan"), 128]], ms.float32)
    net = Net()
    output = net(x)
    expect_output = [[float("nan"), 128.1, -64.45]]
    assert np.allclose(output.asnumpy(), expect_output, equal_nan=True)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_nanmean_int_error(mode):
    """
    Feature: ops.nanmean
    Description: Whether typeerror can be caught if the input is int
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([1, 2, 3], ms.int32)
    net = Net()
    with pytest.raises(TypeError):
        net(x)
        _pynative_executor.sync()
