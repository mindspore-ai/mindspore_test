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
    return np.angle(x_np)


@test_utils.run_with_cell
def forward_func(x):
    return mint.angle(x)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("context_mode", [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_mint_angle_forward(context_mode):
    """
    Feature: pyboost function.
    Description: test function mint.angle forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    real_ = np.array([1, 2, 3, 4])
    imag_ = np.array([4, 5, 6, 7])
    complex_func = np.vectorize(complex)
    x_np = complex_func(real_, imag_).astype(np.complex64)

    out = forward_func(Tensor(x_np))
    expect_out = expect_forward_func(x_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, atol=1e-4, rtol=1e-4)

    x_np = generate_random_input((5, 3), np.float32)
    out = forward_func(Tensor(x_np))
    expect_out = expect_forward_func(x_np)
    np.testing.assert_allclose(out.asnumpy(), expect_out, atol=1e-4, rtol=1e-4)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_mint_angle_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function mint.angle dynamic feature.
    Expectation: expect correct result.
    """
    complex_func = np.vectorize(complex)
    real_1 = np.array([1, 2, 3, 4])
    imag_1 = np.array([4, 5, 6, 7])
    input1 = complex_func(real_1, imag_1).astype(np.complex64)

    real_2 = np.array([[1, 2, 3], [3, 4, 5]])
    imag_2 = np.array([[4, 5, 6], [6, 7, 8]])
    input2 = complex_func(real_2, imag_2).astype(np.complex64)

    TEST_OP(
        forward_func,
        [[Tensor(input1)], [Tensor(input2)]],
        "angle_ext",
        disable_mode=["GRAPH_MODE", "GRAPH_MODE_O0"],
        disable_yaml_check=True,
        disable_grad=True
    )
