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
from mindspore.nn import Cell
import mindspore.ops as ops
from mindspore import context
from mindspore.ops import DataType, CustomRegOp
from tests.mark_utils import arg_mark


def GenerateConcatInput(shape_param):
    np_inputs = []
    tensor_inputs = []
    for shape in shape_param:
        np_input = np.random.rand(*shape).astype(np.float32)
        np_inputs.append(np_input)
        t = ms.Tensor(np_input)
        tensor_inputs.append(t)
    return np_inputs, tensor_inputs


def concat_out_shape(x, y):
    out = x[0].copy()
    out[1] = x[0][1] + x[1][1]
    return out


class CustomNetConcat(Cell):
    def __init__(self, func, out_shape, out_dtype):
        super(CustomNetConcat, self).__init__()
        reg_info = CustomRegOp("aclnnCat") \
            .input(0, "x", "dynamic") \
            .attr("dim", "required", "int") \
            .output(0, "output", "required") \
            .dtype_format(DataType.F16_Default, DataType.F16_Default) \
            .target("Ascend") \
            .get_op_info()

        self.concat = ops.Custom(func, out_shape, out_dtype, func_type="aot", bprop=None,
                                 reg_info=reg_info)

    def construct(self, x, axis):
        res = self.concat(x, axis)
        return res


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_custom_concat_py_infer(context_mode, params):
    """
    Feature: Custom op testcase
    Description: test case for Concat by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=True, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    shape_param, axis = params
    np_inputs, tensor_inputs = GenerateConcatInput(shape_param)
    expected_out = np.concatenate(np_inputs, axis)

    net = CustomNetConcat("aclnnCat", concat_out_shape, lambda x, _: x[0])
    out = net(tensor_inputs, axis)
    assert np.allclose(out.asnumpy(), expected_out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_custom_concat_aclnn_cpp_infer(context_mode, params):
    """
    Feature: Custom op testcase
    Description: test case for Concat by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=True, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    shape_param, axis = params
    np_inputs, tensor_inputs = GenerateConcatInput(shape_param)
    expected_out = np.concatenate(np_inputs, axis)

    net = CustomNetConcat("./infer_file/custom_cpp_infer.cc:aclnnCat", None, None)
    out = net(tensor_inputs, axis)
    assert np.allclose(out.asnumpy(), expected_out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_custom_concat_aclnn_cpp_infer_shape_py_infer_type(context_mode, params):
    """
    Feature: Custom op testcase
    Description: test case for Concat by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=True, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    shape_param, axis = params
    np_inputs, tensor_inputs = GenerateConcatInput(shape_param)
    expected_out = np.concatenate(np_inputs, axis)

    net = CustomNetConcat("./infer_file/custom_cpp_infer.cc:aclnnCat", None, lambda x, _: x[0])
    out = net(tensor_inputs, axis)
    assert np.allclose(out.asnumpy(), expected_out)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE])
@pytest.mark.parametrize("params", [(((2, 2), (2, 3)), 1), (((3, 2, 3), (3, 3, 3)), -2)])
def test_custom_concat_aclnn_cpp_infer_type_py_infer_shape(context_mode, params):
    """
    Feature: Custom op testcase
    Description: test case for Concat by custom
    Expectation: the result match with numpy result
    """

    context.set_context(mode=context_mode, save_graphs=True, save_graphs_path="./graphs",
                        jit_config={"jit_level": "O0"})

    shape_param, axis = params
    np_inputs, tensor_inputs = GenerateConcatInput(shape_param)
    expected_out = np.concatenate(np_inputs, axis)

    net = CustomNetConcat("./infer_file/custom_cpp_infer.cc:aclnnCat", concat_out_shape, None)
    out = net(tensor_inputs, axis)
    assert np.allclose(out.asnumpy(), expected_out)
