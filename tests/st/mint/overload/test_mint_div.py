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
import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import Tensor, mint
from mindspore import context


class NetNone(nn.Cell):
    def construct(self, x, other):
        return mint.div(x, other)


class NetFloor(nn.Cell):
    def construct(self, x, other):
        return mint.div(x, other, rounding_mode="floor")


class NetTrunc(nn.Cell):
    def construct(self, x, other):
        return mint.div(x, other, rounding_mode="trunc")


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_divide_none(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = NetNone()
    x = Tensor(np.array([1.0, 5.0, 7.5]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 3.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.25, 2.5, 2.5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_divide_floor(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide floor
    Expectation: success
    """
    context.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = NetFloor()
    x = Tensor(np.array([1.0, 5.0, 9.5]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 3.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_divide_trunc(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide trunc
    Expectation: success
    """
    context.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = NetTrunc()
    x = Tensor(np.array([1.0, 5.0, 9.5]), mstype.float32)
    y = Tensor(np.array([4.0, 2.0, 3.0]), mstype.float32)
    output = net(x, y)
    expected = np.array([0.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
