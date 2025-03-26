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
from tests.st.ops.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark
from mindspore import Tensor, context
import mindspore as ms

class Net(ms.nn.Cell):
    def construct(self, input1):
        input1 = input1 * 1
        input1.asinh_()
        return input1

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def inplace_asinh_forward_func(input2):
    return Net()(input2)

@ops_binary_cases(OpsBinaryCase(input_info=[((1, 2, 3, 4), np.float32)],
                                output_info=[((1, 2, 3, 4), np.float32)],
                                extra_info='SD5B'))
def ops_asinh_case1(input_binary_data=None, output_binary_data=None):
    output = inplace_asinh_forward_func(Tensor(input_binary_data[0]))
    output_np_fixed = np.nan_to_num(output.asnumpy())
    output_binary_data_fixed = np.nan_to_num(output_binary_data[0])
    assert np.allclose(output_np_fixed, output_binary_data_fixed, 1e-04, 1e-04)

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_inplace_asinh_(mode):
    """
    Feature: tensor.divide()
    Description: Verify the result of tensor.divide
    Expectation: success
    """
    context.set_context(mode=mode)
    ops_asinh_case1()


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_asinh__dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function l1_loss forward with dynamic shape.
    Expectation: expect correct result.
    """
    input3 = ms.Tensor(generate_random_input((7, 8, 9), np.float32))

    input4 = ms.Tensor(generate_random_input((7, 8, 9, 4), np.float32))

    TEST_OP(inplace_asinh_forward_func, [[input3], [input4]], "inplace_asinh",
            disable_mode=['GRAPH_MODE'], inplace_update=True)
