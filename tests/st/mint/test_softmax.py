# Copyright 2025 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, mint
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def expect_forward_output(x_np, dim, dtype=None):
    x_np -= np.max(x_np, axis=dim, keepdims=True)
    out = np.exp(x_np) / np.sum(np.exp(x_np), axis=dim, keepdims=True)
    if dtype is not None:
        out.astype(dtype)
    return out


@test_utils.run_with_cell
def forward_func(x, dim, dtype=None):
    return mint.softmax(x, dim, dtype=dtype)


@test_utils.run_with_cell
def backward_func(x, dim, dtype=None):
    return ms.grad(forward_func, (0))(x, dim, dtype=dtype)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_mint_softmax_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.softmax forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    x_np = generate_random_input((5, 3), np.float32)
    out = forward_func(Tensor(x_np), 0)
    expect_out = expect_forward_output(x_np, 0)
    np.testing.assert_allclose(out.asnumpy(), expect_out, atol=1e-3, rtol=1e-3)

    # Set dtype conversion.
    x_np = generate_random_input((3, 5), np.float32)
    out = forward_func(Tensor(x_np), 1, ms.float16)
    expect_out = expect_forward_output(x_np, 1, np.float16)
    np.testing.assert_allclose(out.asnumpy(), expect_out, atol=1e-3, rtol=1e-3)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_mint_softmax_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.softmax backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    x_np = np.array([-1.1613084, -0.32552388, -0.04373385]).astype(np.float32)
    out_expect = np.array([9.3658e-09, 2.1603e-08, 2.8635e-08]).astype(np.float32)

    out = backward_func(Tensor(x_np), 0)
    np.testing.assert_allclose(out.asnumpy(), out_expect, atol=1e-4, rtol=1e-4)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_mint_softmax_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function mint.softmax dynamic feature.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((3, 5), np.float32)
    input2 = generate_random_input((3, 5, 2, 2), np.float32)

    TEST_OP(
        forward_func,
        [[Tensor(input1), 1], [Tensor(input2), 0]],
        "softmax_ext",
        disable_mode=["GRAPH_MODE"],
        disable_yaml_check=True
    )
