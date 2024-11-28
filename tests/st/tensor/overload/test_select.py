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
"""Test the overload functional method"""
import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import _pynative_executor


class SelectPythonNet(nn.Cell):
    def construct(self, x, condition, y):
        return x.select(condition, y)


class SelectPyboostNet(nn.Cell):
    def construct(self, x, dim, index):
        return x.select(dim, index)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_select_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.select.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    net = SelectPythonNet()
    cond = ms.Tensor([True, False])
    x = ms.Tensor([2, 3], ms.float32)
    y = ms.Tensor([1, 2], ms.float32)
    output = net(x, cond, y)
    expect_output = np.array([2., 2.], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    with pytest.raises(TypeError) as error_info:
        net(x, 1, y)
        _pynative_executor.sync()
    assert "Failed calling select with " in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        temp_y = int(1)
        net(x, cond, temp_y)
        _pynative_executor.sync()
    assert "For 'Tensor.select', if the argument 'y' is int, then the tensor type should be int32 but got " in str(
        error_info.value)

    with pytest.raises(TypeError) as error_info:
        temp_y = float(1)
        x = ms.Tensor([2, 3], ms.int32)
        net(x, cond, temp_y)
        _pynative_executor.sync()
    assert "For 'Tensor.select', if the argument 'y' is float, then the tensor type should be float32 but got " in str(
        error_info.value)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_select_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.select.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    net = SelectPyboostNet()
    x = ms.Tensor([[2, 3, 4, 5], [3, 2, 4, 5]], ms.float32)
    output = net(x, 0, 0)
    expect_output = np.array([2, 3, 4, 5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    output = net(x, 1, 0)
    expect_output = np.array([2, 3], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    output = net(x, 1, 3)
    expect_output = np.array([5, 5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
