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
"""After loading MindIR, should cast scalar dtype from 32bit to 64bit"""

import glob
import pytest
import os
from typing import List

import mindspore
from mindspore import ops, Tensor, nn

from tests.mark_utils import arg_mark
from tests.st.compiler.utils import match_array
from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, _get_current_ir_path


def check_graph_ir_content(ir_name, contents: List[str]):
    ir_files = glob.glob(os.path.join(_get_current_ir_path(), "*" + ir_name + "*.ir"))
    assert len(ir_files) == 1, f'Expect only one {ir_name}_xxx.ir, but {len(ir_files)} found!\n{ir_files}'
    with open(ir_files[0], 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for s in contents:
        found = False
        for line in lines:
            if s in line:
                found = True
                break
        if not found:
            with open(ir_files[0], 'r', encoding='utf-8') as f:
                ir_file_contents = f.read()
            assert False, f'Cannot find "{s}" in {ir_files[0]}! The IR file contents:\n\n{ir_file_contents}'


IR_NAME = 'load_mindir'
BASE_PATH = os.path.realpath(os.path.dirname(__file__))


@save_graph_ir(ir_name=IR_NAME)
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_ops_celu_cast_float_fp32_value_node_to_fp64():
    """
    Feature: Test cast scalar dtype for loaded MindIR.
    Description: For ops.celu(), the inputs are fp64 Tensor and fp32 scalar.
    Expectation: In the dumped graph ir file, the fp32 scalar should be converted to fp64.
    """

    def fn(x: Tensor):
        return ops.celu(x, alpha=0.99)

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    x = ops.randn(2, 2, dtype=mindspore.float32)
    o1 = fn(x)

    mindir_fpath = BASE_PATH + '/exported_mindir/ms_2_5_0/test_ops_celu_cast_float_fp32_value_node_to_fp64.mindir'
    mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_level='O0')
    graph = mindspore.load(mindir_fpath)
    compiled_fn = nn.GraphCell(graph)
    o2 = compiled_fn(x)
    check_graph_ir_content(
        IR_NAME, ['PrimFunc_CeLU(%para1_x, F64(0.99))', '(<Tensor[Float32], (2, 2)>, <Float64, NoShape>)']
    )
    match_array(o1, o2, error=5)


@save_graph_ir(ir_name=IR_NAME)
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_ops_celu_cast_float_fp32_parameter_node_to_fp64():
    """
    Feature: Test cast scalar dtype for loaded MindIR.
    Description: For ops.celu(), the inputs are fp64 Tensor and fp32 scalar.
    Expectation: In the dumped graph ir file, the fp32 scalar should be converted to fp64.
    """

    def fn(x: Tensor, alpha: float):
        return ops.celu(x, alpha=alpha)

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    x = ops.randn(2, 2, dtype=mindspore.float32)
    alpha = mindspore.mutable(0.99)
    o1 = fn(x, alpha)

    mindir_fpath = BASE_PATH + '/exported_mindir/ms_2_5_0/test_ops_celu_cast_float_fp32_parameter_node_to_fp64.mindir'
    mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_level='O0')
    graph = mindspore.load(mindir_fpath)
    compiled_fn = nn.GraphCell(graph)
    o2 = compiled_fn(x, alpha)
    check_contents = [
        'PrimFunc_ScalarCast(%para2_alpha, I64(44))',
        'PrimFunc_CeLU(%para1_x, %0)',
        '(<Tensor[Float32], (2, 2)>, <Float64, NoShape>)',
    ]
    check_graph_ir_content(IR_NAME, check_contents)
    match_array(o1, o2, error=5)


@save_graph_ir(ir_name=IR_NAME)
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_ops_celu_cast_float_fp32_cnode_to_fp64():
    """
    Feature: Test cast scalar dtype for loaded MindIR.
    Description: For ops.celu(), the inputs are fp64 Tensor and fp32 scalar.
    Expectation: In the dumped graph ir file, the fp32 scalar should be converted to fp64.
    """

    def fn(x: Tensor, alpha: float):
        return ops.celu(x, alpha=alpha + 0.01)

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    x = ops.randn(2, 2, dtype=mindspore.float32)
    alpha = mindspore.mutable(0.9)
    o1 = fn(x, alpha)

    mindir_fpath = BASE_PATH + '/exported_mindir/ms_2_5_0/test_ops_celu_cast_float_fp32_cnode_to_fp64.mindir'
    mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_level='O0')
    graph = mindspore.load(mindir_fpath)
    compiled_fn = nn.GraphCell(graph)
    o2 = compiled_fn(x, alpha)
    check_contents = [
        'PrimFunc_ScalarCast(%0, I64(44))',
        '(<Float64, NoShape>, <Int64, NoShape>) -> (<Float64, NoShape>)',
        'PrimFunc_CeLU(%para1_x, %1)',
        '(<Tensor[Float32], (2, 2)>, <Float64, NoShape>)',
    ]
    check_graph_ir_content(IR_NAME, check_contents)
    match_array(o1, o2, error=5)


@save_graph_ir(ir_name=IR_NAME)
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_ops_celu_cast_float_fp32_cnode_to_fp64_v2():
    """
    Feature: Test cast scalar dtype for loaded MindIR.
    Description: For ops.celu(), the inputs are fp64 Tensor and fp32 scalar.
    Expectation: In the dumped graph ir file, the fp32 scalar should be converted to fp64.
    """

    def fn(x: Tensor, y: Tensor):
        alpha = y.nonzero().shape[0] * 0.5
        return ops.celu(x, alpha=alpha)

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    x = ops.randn(2, 2, dtype=mindspore.float32)
    y = mindspore.tensor([1, 0, 1, 0, 1])
    o1 = fn(x, y)

    mindir_fpath = BASE_PATH + '/exported_mindir/ms_2_5_0/test_ops_celu_cast_float_fp32_cnode_to_fp64_v2.mindir'
    mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_level='O0')
    graph = mindspore.load(mindir_fpath)
    compiled_fn = nn.GraphCell(graph)
    o2 = compiled_fn(x, y)
    check_contents = [
        'PrimFunc_ScalarCast(%3, I64(44))',
        '(<Float32, NoShape>, <Int64, NoShape>) -> (<Float64, NoShape>)',
        'PrimFunc_CeLU(%para1_x, %4)',
        '(<Tensor[Float32], (2, 2)>, <Float64, NoShape>)',
    ]
    check_graph_ir_content(IR_NAME, check_contents)
    match_array(o1, o2, error=5)


@save_graph_ir(ir_name=IR_NAME)
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_ops_upsample_cast_tuple_float_fp32_value_node_to_fp64():
    """
    Feature: Test cast scalar dtype for loaded MindIR.
    Description: For ops.upsample(), the inputs are fp64 Tensor and fp32 scalar.
    Expectation: In the dumped graph ir file, the fp32 scalar should be converted to fp64.
    """

    def fn(x: Tensor):
        return ops.upsample(x, scale_factor=(0.5, 1.5, 2.5))

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    x = ops.randn(1, 2, 2, 2, 2, dtype=mindspore.float64)
    o1 = fn(x)

    mindir = BASE_PATH + '/exported_mindir/ms_2_5_0/test_ops_upsample_cast_tuple_float_fp32_value_node_to_fp64.mindir'
    mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_level='O0')
    mindspore.set_context(jit_level='O0')
    graph = mindspore.load(mindir)
    compiled_fn = nn.GraphCell(graph)
    o2 = compiled_fn(x)
    check_contents = [
        'PrimFunc_UpsampleNearest3D(%para1_x, None, (F64(0.5), F64(1.5), F64(2.5)))',
        '(<Tensor[Float64], (1, 2, 2, 2, 2)>, <None, NoShape>, <Tuple[Float64*3]',
    ]
    check_graph_ir_content(IR_NAME, check_contents)
    match_array(o1, o2, error=5)


@pytest.mark.skip(reason="This scene does not exist.")
@save_graph_ir(ir_name=IR_NAME)
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_ops_upsample_cast_tuple_float_fp32_value_node_to_fp64_by_TupleToTensor_and_TensorToTuple():
    """
    Feature: Test cast scalar dtype for loaded MindIR.
    Description: For ops.upsample(), the inputs are fp64 Tensor and fp32 scalar.
    Expectation: In the dumped graph ir file, the fp32 scalar should be converted to fp64.
    """

    def fn(x: Tensor):
        return ops.upsample(x, scale_factor=(0.5, 1.5, 2.5))

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    x = ops.randn(1, 2, 2, 2, 2, dtype=mindspore.float64)
    o1 = fn(x)

    mindir = BASE_PATH + '/exported_mindir/ms_2_5_0/'
    mindir += 'test_ops_upsample_cast_tuple_float_fp32_value_node_to_fp64_by_TupleToTensor_and_TensorToTuple.mindir'
    mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_level='O0')
    mindspore.set_context(jit_level='O0')
    graph = mindspore.load(mindir)
    compiled_fn = nn.GraphCell(graph)
    o2 = compiled_fn(x)
    check_contents = [
        'PrimFunc_TupleToTensor(F32(0.5), F32(1.5), F32(2.5), I64(44))',
        'TensorToTuple(%0)',
        'PrimFunc_UpsampleNearest3D(%para1_x, None, %1)',
    ]
    check_graph_ir_content(IR_NAME, check_contents)
    match_array(o1, o2, error=5)


@save_graph_ir(ir_name=IR_NAME)
@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_ops_log_cast_primitive_attribute_fp32_to_fp64():
    """
    Feature: Test cast scalar dtype for loaded MindIR.
    Description: For ops.log(), the inputs is a fp64 Tensor.
    Expectation: In the dumped graph ir file, the primitive attributes of fp32 dtype should be converted to fp64.
    """

    def fn(x: Tensor):
        return ops.log(x)

    mindspore.set_context(mode=mindspore.PYNATIVE_MODE)
    x = mindspore.tensor([1.5, 2.5, 3.5], dtype=mindspore.float64)
    o1 = fn(x)

    mindir_fpath = BASE_PATH + '/exported_mindir/ms_2_5_0/test_ops_log_cast_primitive_attribute_fp32_to_fp64.mindir'
    mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_level='O0')
    graph = mindspore.load(mindir_fpath)
    compiled_fn = nn.GraphCell(graph)
    o2 = compiled_fn(x)
    check_contents = [
        'PrimFunc_Log(%para1_x) primitive_attrs: {is_load: Bool(1), shift: F64(0), scale: F64(1), '
        'cust_aicpu: "Log", primitive_function: Bool(1), base: F64(-1)}'
    ]
    check_graph_ir_content(IR_NAME, check_contents)
    match_array(o1, o2, error=5)
