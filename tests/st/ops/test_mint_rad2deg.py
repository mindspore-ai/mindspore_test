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


def expect_forward_func(x_np):
    out = np.rad2deg(x_np)
    return out


def expect_backward_func(x_np):
    M_180_PI = 57.295779513082320876798154814105170332405472466564
    dout = np.ones_like(expect_forward_func(x_np))
    dx = dout * M_180_PI
    return dx


@test_utils.run_with_cell
def forward_func(x):
    out = mint.rad2deg(x)
    return out


@test_utils.run_with_cell
def backward_func(x):
    return ms.grad(forward_func, (0))(x)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_mint_rad2deg_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.rad2deg forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    x_np = generate_random_input((3, 3), np.float32)
    out = forward_func(Tensor(x_np))
    expect_out = expect_forward_func(x_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, atol=1e-4, rtol=1e-4)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_mint_rad2deg_backward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.rad2deg backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    x_np = generate_random_input((5, 3), np.float32)
    dx_expect = expect_backward_func(x_np)
    dx = backward_func(Tensor(x_np))
    np.testing.assert_allclose(dx.asnumpy(), dx_expect, atol=1e-4, rtol=1e-4)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_mint_rad2deg_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function rad2deg dynamic feature.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((3, 5), np.float32)
    input2 = generate_random_input((3, 4, 3, 5), np.float32)

    TEST_OP(
        forward_func,
        [[Tensor(input1)], [Tensor(input2)]],
        "rad2deg",
        disable_mode=["GRAPH_MODE"],
        disable_yaml_check=True
    )
