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

import mindspore as ms
from mindspore import JitConfig
from mindspore.nn import Cell
from mindspore.ops.auto_generate.gen_ops_def import add_ext as add
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark
from tests.st.ops.ops_binary_cases import ops_binary_cases, OpsBinaryCase

rtol = 1e-3


class AddCell(Cell):
    def __init__(self):
        super().__init__()
        self.add = add

    def construct(self, x, y, alpha):
        return self.add(x, y, alpha)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_forward(context_mode):
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    add_cell = AddCell()

    x = np.random.randn(1, 16, 4096, 128).astype(np.float32)
    y = np.random.randn(1, 16, 4096, 128).astype(np.float32)
    alpha = 2.0

    output = add_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x + y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    add_cell.set_inputs(ms.tensor(shape=[None, None, None, None], dtype=ms.float16),
                        ms.tensor(shape=[None, None, None, None], dtype=ms.float16), alpha)

    x = np.random.randn(64, 20, 77, 77).astype(np.float16)
    y = np.random.randn(64, 1, 77, 77).astype(np.float16)

    output = add_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x + y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)

    add_cell.set_inputs(ms.tensor(shape=[None, None, None, None], dtype=ms.float16),
                        ms.tensor(shape=None, dtype=ms.float16), alpha)

    x = np.random.randn(3, 73, 3, 768).astype(np.float16)
    y = np.random.randn(1).astype(np.float16)

    output = add_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x + y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_ops_dynamic():
    """
    Feature: ops.extend.add
    Description: dynamic shape and rank
    Expectation: success
    """
    x1 = ms.Tensor(np.array([[1, 2], [3, 4]], np.float32))
    y1 = ms.Tensor(np.array([[5, 6], [7, 8]], np.float32))
    x2 = ms.Tensor(np.array([[1, 2, 3]], np.float32))
    y2 = ms.Tensor(np.array([[10, 11, 12], [13, 14, 15], [16, 17, 18]], np.float32))

    TEST_OP(add, [[x1, y1, 1.], [x2, y2, 2.]], 'add_ext', disable_input_check=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_backward(context_mode):
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    add_cell = AddCell()

    # 2 x 2
    x = np.array([[1, 2], [3, 4]], np.float32)
    y = np.array([[5, 6], [7, 8]], np.float32)
    alpha = 2.0

    output = ms.grad(add_cell, (0))(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = np.ones_like(y)

    np.testing.assert_allclose(output, expect, rtol=rtol)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_bf16(context_mode):
    """
    Feature: ops.extend.add
    Description: bf16
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)

    add_cell = AddCell()

    # 2 x 2
    x = np.array([[1, 2], [3, 4]], np.float32)
    y = np.array([[5, 6], [7, 8]], np.float32)
    alpha = 2.0

    output = ms.grad(add_cell, (0))(ms.tensor(x, ms.bfloat16), ms.tensor(y, ms.bfloat16), alpha).float().asnumpy()
    expect = np.ones_like(y)

    np.testing.assert_allclose(output, expect, rtol=rtol)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_bool(context_mode):
    """
    Feature: test add backward
    Description: test add backward
    Expectation: success
    """
    ms.context.set_context(mode=context_mode)

    add_cell = AddCell()
    add_cell.set_jit_config(JitConfig(jit_level='O0'))

    # 2 x 2
    x = np.array([[True, True], [False, False]], np.bool_)
    y = np.array([[True, False], [True, False]], np.bool_)
    alpha = True

    output = add_cell(ms.tensor(x), ms.tensor(y), alpha).asnumpy()
    expect = x + y * alpha

    np.testing.assert_allclose(output, expect, rtol=rtol)


def ops_add_binary_compare(input_binary_data, output_binary_data):
    add_cell = AddCell()
    output = add_cell(ms.Tensor(input_binary_data[0]), ms.Tensor(input_binary_data[1]), 1.0)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = ms.grad(add_cell, (0))(ms.Tensor(input_binary_data[0]), ms.Tensor(input_binary_data[1]), 1.0)
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)


@ops_binary_cases(OpsBinaryCase(input_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32)],
                                output_info=[((6, 64, 88, 160), np.float32), ((6, 64, 88, 160), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((84, 144, 32), np.float32), ((84, 144, 32), np.float32)],
                                output_info=[((84, 144, 32), np.float32), ((84, 144, 32), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case2(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((1024,), np.float32), ((), np.float32)],
                                output_info=[((1024,), np.float32), ((1024,), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case3(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((48, 32, 32), np.float32), ((), np.float32)],
                                output_info=[((48, 32, 32), np.float32), ((48, 32, 32), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case4(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 6, 288, 64), np.float32), ((), np.float32)],
                                output_info=[((1, 6, 288, 64), np.float32), ((1, 6, 288, 64), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case5(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 1, 1, 288, 64), np.float32), ((), np.float32)],
                                output_info=[((1, 1, 1, 288, 64), np.float32), ((1, 1, 1, 288, 64), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case6(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 576, 128, 16, 2), np.float32), ((1, 576, 128, 16, 1), np.float32)],
                                output_info=[((1, 576, 128, 16, 2), np.float32), ((1, 576, 128, 16, 2), np.float32)],
                                extra_info='auto_drive'))
def ops_add_binary_case7(input_binary_data=None, output_binary_data=None):
    ops_add_binary_compare(input_binary_data, output_binary_data)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_add_binary_cases(mode):
    """
    Feature: Ops
    Description: test op add
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)

    ops_add_binary_case1()
    ops_add_binary_case2()
    ops_add_binary_case3()
    ops_add_binary_case4()
    ops_add_binary_case5()
    ops_add_binary_case6()
    ops_add_binary_case7()
