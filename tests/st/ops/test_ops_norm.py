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

import mindspore as ms
from mindspore import ops, Tensor
from mindspore.ops.function.math_func import norm_ext

import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def vector_norm_forward_func(x, p):
    return ops.vector_norm(x, p)

def vector_norm_backward_func(x, p):
    return ms.grad(vector_norm_forward_func, (0))(x, p)

@test_utils.run_with_cell
def norm_ext_forward_func(x, p):
    return norm_ext(x, p)

@test_utils.run_with_cell
def norm_ext_backward_func(x, p):
    return ms.grad(norm_ext_forward_func, (0))(x, p)

@test_utils.run_with_cell
def norm_ext_forward_dyn(x):
    return norm_ext(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('p', [-np.inf, -1.0, 0, 2.0, 4.0, np.inf])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_norm_forward(mode, p):
    """
    Feature: norm
    Description: Verify the result of norm
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    a = ms.Tensor(np.random.randn(9,), dtype=ms.float32)
    b = a.reshape((3, 3))
    output1 = norm_ext_forward_func(a, p)
    expect_output1 = vector_norm_forward_func(a, p)
    assert np.allclose(output1.asnumpy(), expect_output1.asnumpy())

    output2 = norm_ext_forward_func(b, p)
    expect_output2 = vector_norm_forward_func(b, p)
    assert np.allclose(output2.asnumpy(), expect_output2.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('p', [-np.inf, -1.0, 0, 2.0, 4.0, np.inf])
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_norm_backward(p, mode):
    """
    Feature: norm backward
    Description: Verify the result of norm backward
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    a = ms.Tensor(np.random.randn(9,), dtype=ms.float32)
    b = a.reshape((3, 3))
    output1 = norm_ext_backward_func(a, p)
    expect_output1 = vector_norm_backward_func(a, p)
    assert np.allclose(output1.asnumpy(), expect_output1.asnumpy())

    output2 = norm_ext_backward_func(b, p)
    expect_output2 = vector_norm_backward_func(b, p)
    assert np.allclose(output2.asnumpy(), expect_output2.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_ops_norm_dyn():
    """
    Feature: pyboost function.
    Description: test ops.function.math_func.norm_ext with dynamic rank/shape.
    Expectation: success.
    """
    input_x1 = np.random.randn(*(3, 3)).astype(np.float32)
    input_x2 = np.random.randn(*(3, 3, 3)).astype(np.float32)
    in1 = Tensor(input_x1)
    in2 = Tensor(input_x2)
    TEST_OP(norm_ext_forward_dyn, [[in1], [in2]], '', disable_yaml_check=True, disable_mode=['GRAPH_MODE'])
