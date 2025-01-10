# Copyright 2024 Huawei Technocasties Co., Ltd
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
from mindspore.mint import lerp
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark
import tests.st.utils.test_utils as test_utils


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def lerp_forward_func(x, y, z):
    return lerp(x, y, z)


@test_utils.run_with_cell
def lerp_backward_func(x, y, z):
    return ms.grad(lerp_forward_func, (0, 1, 2))(x, y, z)


def set_mode(mode):
    if mode == "GRAPH_MODE":
        ms.context.set_context(mode=ms.GRAPH_MODE,
                               jit_config={"jit_level": "O0"})
    else:
        ms.context.set_context(mode=ms.PYNATIVE_MODE)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
@pytest.mark.parametrize("context_mode", ["PYNATIVE_MODE", "GRAPH_MODE"])
def test_mint_lerp_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function lerp forward.
    Expectation: expect correct result.
    """
    set_mode(context_mode)
    x = generate_random_input((2, 3, 4), np.float32)
    y = generate_random_input((2, 3, 4), np.float32)
    z = generate_random_input((2, 3, 4), np.float32)
    output = lerp_forward_func(ms.Tensor(x), ms.Tensor(y), ms.Tensor(z))
    backward_output = lerp_backward_func(ms.Tensor(x), ms.Tensor(y),
                                         ms.Tensor(z))
    assert output.asnumpy().dtype == "float32"
    assert output.asnumpy().shape == (2, 3, 4)
    assert backward_output[0].asnumpy().dtype == "float32"
    assert backward_output[0].asnumpy().shape == (2, 3, 4)
    output = lerp_forward_func(ms.Tensor(x), ms.Tensor(y), 0.5)
    backward_output = lerp_backward_func(ms.Tensor(x), ms.Tensor(y), 0.5)
    assert output.asnumpy().dtype == "float32"
    assert output.asnumpy().shape == (2, 3, 4)
    assert backward_output[0].asnumpy().dtype == "float32"
    assert backward_output[0].asnumpy().shape == (2, 3, 4)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential",
)
def test_mint_lerp_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function lerp forward with dynamic shape.
    Expectation: expect correct result.
    """
    input1 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))
    input2 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))
    input3 = ms.Tensor(generate_random_input((2, 3, 4), np.float32))

    input4 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    input5 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    input6 = ms.Tensor(generate_random_input((2, 3, 4, 5), np.float32))
    TEST_OP(lerp_forward_func,
            [[input1, input2, input3], [input4, input5, input6]],
            'lerp',
            disable_yaml_check=True,
            disable_mode=['GRAPH_MODE'])

    TEST_OP(lerp_forward_func, [[input1, input2, 0.5], [input4, input5, 0.7]],
            'lerp',
            disable_yaml_check=True,
            disable_mode=['GRAPH_MODE'])
