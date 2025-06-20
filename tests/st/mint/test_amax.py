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
from mindspore import mint, ops, Tensor
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, dim=(), keepdim=False):
    return mint.max(x, dim=dim, keepdim=keepdim)[0]


def generate_expect_backward_output(x, dim=(), keepdim=False):
    grad = ops.grad(ops.amax)(x, dim, keepdim)
    return grad


@test_utils.run_with_cell
def amax_forward_func(x, dim=(), keepdim=False):
    return mint.amax(x, dim, keepdim)


@test_utils.run_with_cell
def amax_backward_func(x, dim=(), keepdim=False):
    return ms.grad(amax_forward_func, (0))(x, dim, keepdim=keepdim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_amax_normal(mode):
    """
    Feature: pyboost function mint.amax.
    Description: test function amax forward and backward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=mode)
    x = Tensor(generate_random_input((4, 5), np.float32))
    dim = 1
    keepdim = False
    expect = generate_expect_forward_output(x, dim, keepdim)
    expect_grad = generate_expect_backward_output(x, dim, keepdim)
    output = amax_forward_func(x, dim, keepdim)
    output_grad = amax_backward_func(x, dim, keepdim)
    assert np.allclose(output.asnumpy(), expect.asnumpy(), rtol=1.e-5)
    assert np.allclose(output_grad.asnumpy(), expect_grad.asnumpy(), rtol=1.e-5)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_amax_dynamic_shape():
    """
    Feature: Test operator amax.
    Description:  Test operator amax with dynamic input.
    Expectation: the result of amax is correct.
    """
    x1 = Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    x2 = Tensor(generate_random_input((3, 4, 5, 6, 7), np.float32))
    TEST_OP(amax_forward_func, [[x1], [x2]], '', disable_yaml_check=True, disable_mode=["GRAPH_MODE"])
