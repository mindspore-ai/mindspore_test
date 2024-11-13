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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, exponent):
        if isinstance(x, (Tensor)):
            return x.pow(exponent)
        return Tensor.pow(x, exponent)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_pow(mode):
    """
    Feature: Tensor.pow
    Description: Verify the result of Tensor.pow
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.array([1.0, 2.0, 4.0]), ms.float32)
    exponent = 3.0
    net = Net()
    output_x = net(x, exponent)
    expect_x = Tensor(np.array([1.0, 8.0, 64.0]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    exponent = Tensor(np.array([2.0, 4.0, 3.0]), ms.float32)
    net = Net()
    output_x = net(x, exponent)
    expect_x = Tensor(np.array([1.0, 16.0, 64.0]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_pow_scalar(mode):
    """
    Feature: Tensor.pow
    Description: Verify the result of Tensor.pow
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = 2.0
    exponent = 3.0
    net = Net()
    output_x = net(x, exponent)
    expect_x = 8.0
    assert output_x == expect_x

    exponent = Tensor(np.array([2.0, 4.0, 3.0]), ms.float32)
    net = Net()
    output_x = net(x, exponent)
    expect_x = Tensor(np.array([4.0, 16.0, 8.0]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())
