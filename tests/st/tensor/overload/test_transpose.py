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
import mindspore as ms
import mindspore.nn as nn
from mindspore.common.api import _pynative_executor

from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils


class TransposePythonNet(nn.Cell):
    def construct(self, x, axes):
        return x.transpose(axes)


class TransposePyboostNet(nn.Cell):
    def construct(self, x, dim0, dim1):
        return x.transpose(dim0, dim1)


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def transpose_ext_forward_func(x, dim0, dim1):
    return x.transpose(dim0, dim1)


@test_utils.run_with_cell
def transpose_forward_func(x, axes):
    return x.transpose(axes)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_transpose_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.transpose.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = TransposePythonNet()

    x = ms.Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), ms.float32)
    axes = (0, 2, 1)
    output = net(x, axes)
    expect_output = np.array([[[1., 4.],
                               [2., 5.],
                               [3., 6.]],
                              [[7., 10.],
                               [8., 11.],
                               [9., 12.]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    # out of range [-rank(input), rank(input))
    with pytest.raises(ValueError) as error_info:
        axes = (0, 3, 4)
        net(x, axes)
        _pynative_executor.sync()
    if mode == 0:
        assert "the perm value must be in" in str(error_info.value)
    elif mode == 1:
        assert "dim value error." in str(error_info.value)
    # axes's shape not equal to input's shape
    with pytest.raises(ValueError) as error_info:
        axes = (0, 1)
        net(x, axes)
        _pynative_executor.sync()
    assert "size of perm should equal to rank of " in str(error_info.value)
    # axes have same element
    with pytest.raises(ValueError) as error_info:
        axes = (0, 1, 1)
        net(x, axes)
        _pynative_executor.sync()
    assert "perms should all be unique dim" in str(error_info.value)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_transpose_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.transpose.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = TransposePyboostNet()

    x = ms.Tensor(np.ones((2, 3, 4), dtype=np.float32))
    output = net(x, 0, 2)
    expect_output = (4, 3, 2)
    assert np.allclose(output.shape, expect_output)

    with pytest.raises(ValueError) as error_info:
        net(x, 0, float(2.0))
        _pynative_executor.sync()
    assert "the number of axes must be equal to the dimension of Tensor" in str(error_info.value)

    with pytest.raises(ValueError) as error_info:
        net(x, 0, 3)
        _pynative_executor.sync()
    assert "For primitive[TransposeExt], the dim1 must be in " in str(error_info.value)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
def test_tensor_transpose_dynamic():
    """
    Feature: Test transpose op.
    Description: Test transpose dynamic shape.
    Expectation: the result match with expected result.
    """
    ms_data1 = ms.Tensor(generate_random_input((4, 6), np.float32))
    dim0_1 = 0
    dim1_1 = 1
    ms_data2 = ms.Tensor(generate_random_input((5, 2, 7, 3), np.float32))
    dim0_2 = 1
    dim1_2 = 2
    if ms.get_context('device_target') != 'Ascend':
        with pytest.raises(RuntimeError) as error_info:
            TEST_OP(transpose_ext_forward_func, [[ms_data1, dim0_1, dim1_1], [ms_data2, dim0_2, dim1_2]],
                    'transpose_ext', disable_mode=['GRAPH_MODE'])
        assert "Unsupported op [TransposeExt] on " in str(error_info.value)

    axes1 = (0, 1)
    axes2 = (0, 2, 1, 3)
    TEST_OP(transpose_forward_func, [[ms_data1, axes1], [ms_data2, axes2]], 'transpose', disable_yaml_check=True,
            disable_mode=['GRAPH_MODE'])
