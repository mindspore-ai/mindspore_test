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
import random

import mindspore as ms
from mindspore import ops, Tensor
from mindspore.ops.function.array_func import fill_
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


@run_with_cell
def fill__forward_func(input_x, vec2):
    return fill_(input_x, vec2)


@arg_mark(
    plat_marks=["platform_ascend", "platform_ascend910b"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("mode", ["pynative"])
def test_ops_fill__normal(mode):
    """
    Feature: ops.function.array_func.fill_
    Description: Verify the result of ops.function.array_func.fill_
    Expectation: success
    """
    input_x = ops.full((100, 100), 0, dtype=ms.float32)
    value = 10
    except_out = ops.full((100, 100), 10, dtype=ms.float32)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output1 = fill__forward_func(input_x, value)
        output2 = fill__forward_func(input_x, Tensor(value))
    allclose_nparray(output1.asnumpy(), except_out.asnumpy(), 1e-04, 1e-04)
    allclose_nparray(output2.asnumpy(), except_out.asnumpy(), 1e-04, 1e-04)


@arg_mark(
    plat_marks=["platform_ascend", "platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_ops_fill_tensor_dynamic():
    """
    Feature: Test outer with dynamic shape in graph mode using TEST_OP.
    Description: call ops.function.array_func.fill_ with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3), np.float32)
    y1 = random.random()
    x2 = generate_random_input((4, 5, 6), np.float32)
    y2 = random.random()
    TEST_OP(
        fill_,
        [[Tensor(x1), Tensor(y1)], [Tensor(x2), Tensor(y2)]],
        "inplace_fill_tensor",
        disable_mode=["GRAPH_MODE", "GRAPH_MODE_O0"],
        disable_input_check=True,
        disable_grad=True,
        inplace_update=True
    )


@arg_mark(
    plat_marks=["platform_ascend", "platform_ascend910b"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_ops_fill_scalar_dynamic():
    """
    Feature: Test outer with dynamic shape in graph mode using TEST_OP.
    Description: call ops.function.array_func.fill_ with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3), np.float32)
    y1 = random.random()
    x2 = generate_random_input((4, 5, 6), np.float32)
    y2 = random.random()
    TEST_OP(
        fill_,
        [[Tensor(x1), y1], [Tensor(x2), y2]],
        "inplace_fill_tensor",
        disable_mode=["GRAPH_MODE", "GRAPH_MODE_O0"],
        disable_input_check=True,
        disable_grad=True,
        inplace_update=True
    )
