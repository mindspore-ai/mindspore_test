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
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils

import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import _pynative_executor


class MinPythonNet(nn.Cell):
    def construct(self, x, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False):
        return x.min(axis, keepdims, initial=initial, where=where, return_indices=return_indices)


class MinPyboostNet(nn.Cell):
    def construct(self, x):
        return x.min()


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def min_forward_func1(x):
    return x.min()


@test_utils.run_with_cell
def min_forward_func2(x, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False):
    return x.min(axis, keepdims, initial=initial, where=where, return_indices=return_indices)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_min_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.min.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    # test 1: using positional args
    net = MinPythonNet()
    x = ms.Tensor(np.arange(4).reshape((2, 2)).astype(np.float32))
    output = net(x, 0, False, initial=9, where=ms.Tensor([False, True]), return_indices=False)
    expect_output = np.array([9., 1.], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, (0, 1), False, initial=9, where=ms.Tensor([False, True]), return_indices=False)
    expect_output = 1.0
    assert np.allclose(output.asnumpy(), expect_output)

    # test 2: using default args.
    output = net(x)
    assert np.allclose(output.asnumpy(), 0.0)

    # test 3: using k-v args.
    output = net(x, axis=0, keepdims=False, initial=9, where=ms.Tensor([False, True]), return_indices=False)
    expect_output = np.array([9., 1.], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    # test 4: error input.
    with pytest.raises(TypeError) as error_info:
        net(x, axis=0, keepdims=False, initial=9, where=None, return_indices=False)
        _pynative_executor.sync()
    assert "Failed calling min with " in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        net(x, axis=0, keepdims=False, initial=ms.Tensor([False, True]), where=ms.Tensor([False, True]),
            return_indices=False)
        _pynative_executor.sync()
    assert "Failed calling min with " in str(error_info.value)

    with pytest.raises(TypeError) as error_info:
        net(x, axis=0, keepdims=False, initial=9, where=ms.Tensor([False, True]), return_indices=1)
        _pynative_executor.sync()
    assert "Failed calling min with " in str(error_info.value)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_min_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.min.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = MinPyboostNet()
    x = ms.Tensor(np.arange(4).reshape((2, 2)).astype(np.float32))
    output = net(x)
    assert np.allclose(output.asnumpy(), 0.0)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_tensor_min_dynamic():
    """
    Feature: Test min op.
    Description: Test min dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 3, 6), np.float32))
    axis1 = None
    keepdims1 = False
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    axis2 = None
    keepdims2 = False
    TEST_OP(min_forward_func1, [[ms_data1, axis1], [ms_data2]],
            'min', disable_mode=['GRAPH_MODE'])
    TEST_OP(min_forward_func2, [[ms_data1, axis1, keepdims1], [ms_data2, axis2, keepdims2]], 'min',
            disable_mode=['GRAPH_MODE'], disable_input_check=True, disable_yaml_check=True)
