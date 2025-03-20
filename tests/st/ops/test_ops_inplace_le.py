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
import pytest
import numpy as np
import mindspore as ms
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark

class Net(ms.nn.Cell):
    def construct(self, input1, other1):
        input1.le_(other1)
        return input1

class Net1(ms.nn.Cell):
    def construct(self, input2, other2):
        input2 = input2.clone()
        input2.le_(other2)
        return input2

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_output(input3, other3):
    return np.less_equal(input3, other3)

def inplace_le_forward_func(input4, other4):
    return Net()(input4, other4)

def inplace_le_dyn_func(input5, other5):
    return Net1()(input5, other5)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_ops_inplace_le(context_mode):
    """
    Feature: pyboost function.
    Description: test function round forward and backward on Ascend.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5), np.float32)
    expect = generate_expect_output(x, y)
    input6 = ms.Tensor(x)
    other6 = ms.Tensor(y)
    output = inplace_le_forward_func(input6, other6)
    np.testing.assert_allclose(output.asnumpy(), expect)
    np.testing.assert_allclose(input6.asnumpy(), expect)

    x1 = np.array([0.2, -0.5, 0.6, 1.8])
    y1 = 0.5
    expect1 = generate_expect_output(x1, y1)
    input7 = ms.Tensor(x1)
    other7 = y1
    output = inplace_le_forward_func(input7, other7)
    np.testing.assert_allclose(output.asnumpy(), expect1)
    np.testing.assert_allclose(input7.asnumpy(), expect1)

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_ops_inplace_le_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function l1_loss forward with dynamic shape.
    Expectation: expect correct result.
    """
    input8 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    other8 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    input9 = ms.Tensor(generate_random_input((7, 8, 9, 4), np.float32))
    other9 = ms.Tensor(generate_random_input((7, 8, 9, 4), np.float32))

    TEST_OP(inplace_le_dyn_func, [[input8, other8], [input9, other9]], "inplace_le_tensor",
            disable_input_check=True, disable_mode=['GRAPH_MODE'], inplace_update=True)

    input10 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))
    other10 = 0.03

    input11 = ms.Tensor(generate_random_input((7, 8, 9, 4), np.float32))
    other11 = -0.01

    TEST_OP(inplace_le_dyn_func, [[input10, other10], [input11, other11]], "inplace_le_scalar",
            disable_input_check=True, disable_mode=['GRAPH_MODE'], inplace_update=True)
