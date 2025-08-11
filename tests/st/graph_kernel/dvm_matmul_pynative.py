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

import numpy as np
import mindspore.ops as ops
import mindspore.context as context
from mindspore import mint
from tests.st.graph_kernel.gk_utils import gen_flag, gen_input, get_func_name, compare_outputs

np.random.seed(1)
context.set_context(mode=context.PYNATIVE_MODE)


def test_dense():
    """
    Feature: Dense
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(shape1, shape2, shape_bias, data_type):
        flag = gen_flag("dense", shape1, shape2, shape_bias, data_type)
        x0 = gen_input(shape1, data_type, data_range=0.01)
        x1 = gen_input(shape2, data_type, data_range=0.01)
        x2 = gen_input(shape_bias, data_type, data_range=0.01) if shape_bias else None
        y0 = ops.auto_generate.Dense()(x0, x1, x2)
        compare_outputs(flag, y0, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    for d in ["float16", "bfloat16"]:
        _run((4096, 7168), (2112, 7168), (2112,), d)
        _run((4096, 7168), (2112, 7168), None, d)


def test_matmul():
    """
    Feature: MatMul, BatchMatMul
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, shape1, shape2, data_type, transpose_a, transpose_b):
        flag = gen_flag(get_func_name(func), shape1, shape2, data_type, transpose_a, transpose_b)
        x0 = gen_input(shape1, data_type, data_range=0.01)
        x1 = gen_input(shape2, data_type, data_range=0.01)
        y0 = func(transpose_a, transpose_b)(x0, x1)
        compare_outputs(flag, y0, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    _run(ops.auto_generate.MatMul, (4096, 1536), (16384, 1536), "float16", False, True)
    _run(ops.auto_generate.MatMul, (4096, 1536), (1536, 8192), "bfloat16", False, False)
    _run(ops.auto_generate.BatchMatMul, (4, 4096, 1536), (1, 8192, 1536), "float16", False, True)
    _run(ops.auto_generate.BatchMatMul, (4096, 1024), (4, 8, 1024, 1024), "bfloat16", False, True)


def test_matmul_ext():
    """
    Feature: MatMulExt, BatchMatMulExt
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run(func, shape1, shape2, data_type):
        flag = gen_flag(get_func_name(func), shape1, shape2, data_type)
        x0 = gen_input(shape1, data_type, data_range=0.01)
        x1 = gen_input(shape2, data_type, data_range=0.01)
        y0 = func(x0, x1)
        compare_outputs(flag, y0, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    _run(mint.matmul, (2, 4096, 1024), (1024, 2048), "float16")
    _run(mint.matmul, (4096, 1024), (4, 1024, 2048), "bfloat16")
    _run(ops.auto_generate.BatchMatMulExt(), (4, 4096, 7168), (4, 7168, 2112), "float16")
    _run(ops.auto_generate.BatchMatMulExt(), (4, 4096, 7168), (4, 7168, 2112), "bfloat16")


def test_matmul_ext_view():
    """
    Feature: view + matmul
    Description: pynative mode
    Expectation: the result match with the expected result
    """

    def _run_transpose(func, shape1, shape2, data_type, trans_axis):
        flag = gen_flag(get_func_name(func), "transpose", shape1, shape2, data_type, trans_axis)
        x0 = gen_input(shape1, data_type, data_range=0.01)
        x1 = gen_input(shape2, data_type, data_range=0.01)
        y0 = func(x0, ops.Transpose()(x1, trans_axis))
        compare_outputs(flag, y0, cmp_precision=4e-3 if data_type == "bfloat16" else 1e-3)

    _run_transpose(mint.matmul, (2, 4096, 1024), (2048, 1024), "float16", (1, 0))
    _run_transpose(mint.matmul, (4096, 1024), (4, 2048, 1024), "float16", (0, 2, 1))
    _run_transpose(ops.auto_generate.BatchMatMulExt(), (4, 4096, 7168), (7168, 4, 2112), "float16", (1, 0, 2))
