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
from mindspore import Tensor

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_logical_and(mode):
    """
    Feature: test Tensor.logical_and.
    Description: Verify the result of Tensor.logical_and..
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x_np = np.array([True, False, True]).astype(np.bool_)
    y_np = np.array([True, True, False]).astype(np.bool_)
    x = Tensor(x_np, dtype=ms.bool_)
    y = Tensor(y_np, dtype=ms.bool_)
    expect_output = np.logical_and(x_np, y_np)
    output = x.logical_and(y)
    assert np.allclose(output.asnumpy(), expect_output)

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_logical_not(mode):
    """
    Feature: test Tensor.logical_not.
    Description: Verify the result of Tensor.logical_not..
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x_np = np.array([True, False, True]).astype(np.bool_)
    x = Tensor(x_np, dtype=ms.bool_)
    expect_output = np.logical_not(x_np)
    output = x.logical_not()
    assert np.allclose(output.asnumpy(), expect_output)

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_logical_or(mode):
    """
    Feature: test Tensor.logical_or.
    Description: Verify the result of Tensor.logical_or..
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x_np = np.array([True, False, True]).astype(np.bool_)
    y_np = np.array([True, True, False]).astype(np.bool_)
    x = Tensor(x_np, dtype=ms.bool_)
    y = Tensor(y_np, dtype=ms.bool_)
    expect_output = np.logical_or(x_np, y_np)
    output = x.logical_or(y)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_logical_and_graph_mode():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.logical_and.
    Expectation: Run success
    """
    @ms.jit
    def func_and(x, other):  # pylint: disable=redefined-builtin
        return x.logical_and(other)

    x_np = np.array([True, False, False]).astype(np.bool_)
    y_np = np.array([False, True, False]).astype(np.bool_)
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = func_and(x, y)
    expect = np.logical_and(x_np, y_np)
    assert np.all(out.asnumpy() == expect)

@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_logical_or_graph_mode():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.logical_or.
    Expectation: Run success
    """
    @ms.jit
    def func_or(x, other):  # pylint: disable=redefined-builtin
        return x.logical_or(other)

    x_np = np.array([True, False, False]).astype(np.bool_)
    y_np = np.array([False, True, False]).astype(np.bool_)
    x = ms.Tensor(x_np)
    y = ms.Tensor(y_np)
    out = func_or(x, y)
    expect = np.logical_or(x_np, y_np)
    assert np.all(out.asnumpy() == expect)

@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_logical_not_graph_mode():
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.logical_not.
    Expectation: Run success
    """
    @ms.jit
    def func_not(x):  # pylint: disable=redefined-builtin
        return x.logical_not()

    x_np = np.array([True, False, False]).astype(np.bool_)
    x = ms.Tensor(x_np)
    out = func_not(x)
    expect = np.logical_not(x_np)
    assert np.all(out.asnumpy() == expect)
