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
from mindspore import ops, mint, Tensor, jit, context
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    x = np.random.randn(*shape).astype(dtype)
    y = np.random.randn(*shape).astype(dtype)
    expect = np.bitwise_or(x, y)
    return x, y, expect


@test_utils.run_with_cell
def bitwise_or_forward_func(x, y):
    return mint.bitwise_or(x, y)


@test_utils.run_with_cell
def bitwise_or_backward_func(x, y):
    return ops.grad(bitwise_or_forward_func, 0)(x, y)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", ['pynative', 'KBK'])
def test_bitwise_or_forward_backward(mode):
    """
    Feature: pyboost function.
    Description: test function bitwise_or forward.
    Expectation: expect correct result.
    """
    x, y, expect = generate_random_input((2, 3, 4, 5), np.int32)
    y2 = 6
    expect2 = np.bitwise_or(x, y2)
    x = Tensor(x, dtype=ms.int32)
    y = Tensor(y, dtype=ms.int32)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = bitwise_or_forward_func(x, y)
        output2 = bitwise_or_forward_func(x, y2)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(bitwise_or_forward_func, backend="ms_backend", jit_level="O0"))(x, y)
        output2 = (jit(bitwise_or_forward_func, backend="ms_backend", jit_level="O0"))(x, y2)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = bitwise_or_forward_func(x, y)
        output2 = bitwise_or_forward_func(x, y2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-3)

    x, y, _ = generate_random_input((2, 3, 4, 5), np.int32)
    expect = np.zeros((2, 3, 4, 5))
    y2 = 6
    x = Tensor(x, dtype=ms.int32)
    y = Tensor(y, dtype=ms.int32)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        grad = bitwise_or_backward_func(x, y)
        grad2 = bitwise_or_backward_func(x, y2)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        grad = (jit(bitwise_or_backward_func, backend="ms_backend", jit_level="O0"))(x, y)
        grad2 = (jit(bitwise_or_backward_func, backend="ms_backend", jit_level="O0"))(x, y2)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        grad = bitwise_or_backward_func(x, y)
        grad2 = bitwise_or_backward_func(x, y2)
    assert np.allclose(grad.asnumpy(), expect)
    assert np.allclose(grad2.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_bitwise_or_dynamic_shape_scalar():
    """
    Feature: Test bitwise_or op.
    Description: Test bitwise_or dynamic shape.
    Expectation: the result match with expected result.
    """
    x, y, _ = generate_random_input((3, 4, 5, 6), np.int32)
    x = Tensor(x, dtype=ms.int64)
    y = 6
    x2, y2, _ = generate_random_input((3, 4), np.int64)
    x2 = Tensor(x2, dtype=ms.int64)
    y2 = 3
    TEST_OP(bitwise_or_forward_func, [[x, y], [x2, y2]], '',
            disable_yaml_check=True, disable_mode=['GRAPH_MODE'])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_bitwise_or_dynamic_shape_tensor():
    """
    Feature: Test bitwise_or op.
    Description: Test bitwise_or dynamic shape.
    Expectation: the result match with expected result.
    """
    x, y, _ = generate_random_input((3, 4, 5, 6), np.int32)
    x = Tensor(x, dtype=ms.int64)
    y = Tensor(y, dtype=ms.int64)
    x2, y2, _ = generate_random_input((3, 4), np.int64)
    x2 = Tensor(x2, dtype=ms.int64)
    y2 = Tensor(y2, dtype=ms.int64)
    TEST_OP(bitwise_or_forward_func, [[x, y], [x2, y2]], '',
            disable_yaml_check=True, disable_mode=['GRAPH_MODE'])
