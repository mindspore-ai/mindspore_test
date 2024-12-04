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
import mindspore as ms
import mindspore.nn as nn
import numpy as np
import pytest

from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils


class SumPythonNet(nn.Cell):
    def construct(self, x, axis=None, dtype=None, keepdims=False, initial=None):
        return x.sum(axis, dtype, keepdims, initial)


class SumPyboostNet(nn.Cell):
    def construct(self, x, dim=None, keepdim=False, *, dtype=None):
        return x.sum(dim, keepdim, dtype)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def sum_ext_forward_func(x, dim=None, keepdim=False, dtype=None):
    return x.sum(dim, keepdim, dtype=dtype)


@test_utils.run_with_cell
def sum_forward_func(x, axis=None, dtype=None, keepdims=False, initial=None):
    return x.sum(axis, dtype, keepdims, initial)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_sum_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.sum.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = SumPythonNet()

    x = ms.Tensor([[[1, 2, 3], [2, 3, 4]]], ms.float32)
    output = net(x)
    expect_output = 15.0
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, axis=[0, 1], keepdims=True)
    expect_output = np.array([[[3., 5., 7.]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, axis=(2,), keepdims=False)
    expect_output = np.array([[6., 9.]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, axis=2, keepdims=True)
    expect_output = np.array([[[6.], [9.]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, dtype=ms.bool_, initial=12)
    expect_output = True
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_sum_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.sum.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = SumPyboostNet()

    x = ms.Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                            [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                            [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), ms.float32)
    output = net(x)
    expect_output = 270.0
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, dim=2)
    expect_output = np.array([[6., 12., 18.],
                              [24., 30., 36.],
                              [42., 48., 54.]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, dim=2, keepdim=True)
    expect_output = np.array([[[6.],
                               [12.],
                               [18.]],
                              [[24.],
                               [30.],
                               [36.]],
                              [[42.],
                               [48.],
                               [54.]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_tensor_sum_dynamic():
    """
    Feature: Test sum op.
    Description: Test sum dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 6), np.float32))
    dim1 = 1
    keepdim1 = False
    dtype1 = None
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    dim2 = 2
    keepdim2 = True
    dtype2 = None
    TEST_OP(sum_ext_forward_func,
            [[ms_data1, dim1, keepdim1, dtype1], [ms_data2, dim2, keepdim2, dtype2]], 'sum_ext',
            disable_mode=['GRAPH_MODE'], disable_input_check=True, disable_nontensor_dynamic_type='BOTH')

    ms_data1 = ms.Tensor(generate_random_input((2, 6), np.float32))
    axis1 = 1
    dtype1 = None
    keepdims1 = True
    initial1 = 3
    ms_data2 = ms.Tensor(generate_random_input((3, 2, 7, 3), np.float32))
    axis2 = 2
    dtype2 = None
    keepdims2 = False
    initial2 = 2
    TEST_OP(sum_forward_func,
            [[ms_data1, axis1, dtype1, keepdims1, initial1], [ms_data2, axis2, dtype2, keepdims2, initial2]], 'sum',
            disable_mode=['GRAPH_MODE'], disable_input_check=True, disable_yaml_check=True,
            disable_nontensor_dynamic_type='BOTH')
