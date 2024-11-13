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
import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):

    def construct(self, x, other):
        return x.not_equal(other)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_not_equal_tensor_scalar(mode):
    """
    Feature: tensor.not_equal
    Description: Verify the result of tensor.not_equal
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor([1, 2, 3], ms.float32)
    other = 2.0
    net = Net()
    output_x = net(x, other)
    expect_x = Tensor(np.array([True, False, True]))
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    x = Tensor([True, True, False])
    other = False
    net = Net()
    output_x = net(x, other)
    expect_x = Tensor(np.array([True, True, False]))
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_not_equal_tensor_tensor(mode):
    """
    Feature: tensor.not_equal
    Description: Verify the result of tensor.not_equal
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor([1, 2, 3], ms.int32)
    other = Tensor([1, 2, 4], ms.int32)
    net = Net()
    output_x = net(x, other)
    expect_x = Tensor(np.array([False, False, True]))
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    x = Tensor([True, True, False])
    other = Tensor([False, True, True])
    net = Net()
    output_x = net(x, other)
    expect_x = Tensor(np.array([True, False, True]))
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())
