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
from tests.st.ops.ops_binary_cases import ops_binary_cases, OpsBinaryCase
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils
import numpy as np
import mindspore as ms
from mindspore import Tensor, context

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@test_utils.run_with_cell
def index_put_forward_func(input1, indices1, values1, accumulate1):
    return input1.index_put(indices1, values1, accumulate1)

@test_utils.run_with_cell
def index_put_backward_func(input2, indices2, values2, accumulate2):
    return ms.grad(index_put_forward_func, (0, 2))(input2, indices2, values2, accumulate2)

@test_utils.run_with_cell
def index_put_dyn_func(input3, indices3, values3, accumulate3):
    return input3.index_put(indices3, values3, accumulate3)

@ops_binary_cases(OpsBinaryCase(input_info=[((1, 2, 3, 4), np.float32)],
                                output_info=[((1, 2, 3, 4), np.float32), ((1, 2, 3, 4), np.float32),
                                             ((3,), np.float32)],
                                extra_info='SD5B'))
def ops_index_put_case1(input_binary_data=None, output_binary_data=None):
    input4 = Tensor(input_binary_data[0])
    indices4 = [Tensor([0, 0, 0]), Tensor([0, 1, 0]), Tensor([0, 1, 2]), Tensor([1, 2, 3])]
    values4 = Tensor([1.1, 2.7, 3.8], dtype=ms.float32)
    accumulate4 = False
    output = index_put_forward_func(input4, indices4, values4, accumulate4)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    grad_output = index_put_backward_func(input4, indices4, values4, accumulate4)
    assert np.allclose(grad_output[0].asnumpy(), output_binary_data[1], 1e-04, 1e-04)
    assert np.allclose(grad_output[1].asnumpy(), output_binary_data[2], 1e-04, 1e-04)

@ops_binary_cases(OpsBinaryCase(input_info=[((3, 4, 6, 8, 7), np.float32)],
                                output_info=[((3, 4, 6, 8, 7), np.float32), ((3, 4, 6, 8, 7), np.float32),
                                             ((3,), np.float32)],
                                extra_info='SD5B'))
def ops_index_put_case2(input_binary_data=None, output_binary_data=None):
    input5 = Tensor(input_binary_data[0])
    indices5 = [Tensor([1, 2, 2]), Tensor([1, 2, 2]), Tensor([2, 3, 4]), Tensor([2, 3, 4]), Tensor([3, 4, 5])]
    values5 = Tensor(np.array([2.7, 4.2, 1.1]), dtype=ms.float32)
    accumulate5 = True
    output = index_put_forward_func(input5, indices5, values5, accumulate5)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    grad_output = index_put_backward_func(input5, indices5, values5, accumulate5)
    assert np.allclose(grad_output[0].asnumpy(), output_binary_data[1], 1e-04, 1e-04)
    assert np.allclose(grad_output[1].asnumpy(), output_binary_data[2], 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_index_put_binary_cases(context_mode):
    """
    Feature: Ops
    Description: test op rotated_iou pynative
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode, save_graphs=True, save_graphs_path="./ir_dump")

    ops_index_put_case1()
    ops_index_put_case2()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_ops_index_put_binary_cases_GE():
    """
    Feature: Ops
    Description: test op rotated_iou pynative
    Expectation: expect correct result.
    """
    context.set_context(mode=ms.GRAPH_MODE, jit_level='O2', device_target="Ascend")
    context.set_context(save_graphs=True, save_graphs_path="./ir_dump")
    ops_index_put_case1()
    ops_index_put_case2()

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_ops_index_put_dynamic_shape_ascend():
    """
    Feature: pyboost function.
    Description: test function round with dynamic shape on Ascend.
    Expectation: expect correct result.
    """
    input6 = generate_random_input((1, 2, 3, 4), np.float32)
    indices6 = [Tensor([0]), Tensor([0]), Tensor([0]), Tensor([0])]
    values6 = Tensor(np.array([1.1]), dtype=ms.float32)
    accumulate6 = False

    input7 = generate_random_input((2, 2, 6, 7, 4), np.float32)
    indices7 = [Tensor([0, 1]), Tensor([0, 1]), Tensor([0, 1]), Tensor([0, 1]), Tensor([0, 1])]
    values7 = Tensor(np.array([1.1, 2.3]), dtype=ms.float32)
    accumulate7 = True
    TEST_OP(index_put_dyn_func, [[ms.Tensor(input6), indices6, values6, accumulate6],
                                 [ms.Tensor(input7), indices7, values7, accumulate7]],
            'index_put', disable_yaml_check=True, disable_input_check=True,
            disable_mode=['GRAPH_MODE', 'GRAPH_MODE_O0'])
