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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, jit
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


class Net(ms.nn.Cell):
    def construct(self, x, y):
        z = x * 1
        return z.copy_(y)


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def copy_forward_func(x, y):
    return Net()(x, y)


def copy_backward_func(x, y):
    return ops.grad(copy_forward_func, grad_position=(0, 1))(x, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_copy_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function copy.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 2, 3, 4), np.float32)
    y = generate_random_input((2, 2, 3, 4), np.float32)
    z = generate_random_input((2, 1, 4), np.float32)  # broadcast

    expect_y_grad = np.ones_like(y, dtype=np.float32)
    expect_z = np.expand_dims(z.repeat(3, axis=1), axis=0).repeat(2, axis=0)
    expect_z_grad = np.ones_like(z, dtype=np.float32) * 6

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output_y = copy_forward_func(ms.Tensor(x), ms.Tensor(y))
        output_y_grad = copy_backward_func(ms.Tensor(x), ms.Tensor(y))

        output_z = copy_forward_func(ms.Tensor(x), ms.Tensor(z))
        output_z_grad = copy_backward_func(ms.Tensor(x), ms.Tensor(z))
    else:
        output_y = (jit(copy_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(y))
        output_y_grad = (jit(copy_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(y))

        output_z = (jit(copy_forward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(z))
        output_z_grad = (jit(copy_backward_func, backend="ms_backend", jit_level="O0"))(ms.Tensor(x), ms.Tensor(z))
    np.allclose(output_y.asnumpy(), y, rtol=1e-5, equal_nan=True)
    np.allclose(output_y_grad[1].asnumpy(), expect_y_grad, rtol=1e-5, equal_nan=True)

    np.allclose(output_z.asnumpy(), expect_z, rtol=1e-5, equal_nan=True)
    np.allclose(output_z_grad[1].asnumpy(), expect_z_grad, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_copy_cpu_inplace_feature():
    """
    Feature: inplace features.
    Description: test copy forward .
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    expect = np.array([0, 6, 5, 6]).astype(np.int64).reshape((2, 2))
    inputs = [
        [ms.Tensor([[1, 2], [3, 4]]), ms.Tensor([[5, 6], [5, 6]])],
        [ms.Tensor([[1, 2], [3, 4]]), ms.Tensor([[5., 6.], [5., 6.]])],
        [ms.Tensor([[1, 2], [3, 4]]), ms.Tensor([5, 6])],
        [ms.Tensor([[1, 2], [3, 4]]), ms.Tensor([5., 6.])],
    ]
    for param in inputs:
        x = param[0]
        y = param[1]
        z = x.copy_(y)
        x[0][0] = 0
        np.testing.assert_equal(expect, x.asnumpy())
        np.testing.assert_equal(expect, z.asnumpy())


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
def test_copy_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test copy forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_y1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    tensor_y2 = ms.Tensor(generate_random_input((1, 1, 5), np.float32))  # broadcast

    TEST_OP(copy_forward_func, [[tensor_x1, tensor_y1], [tensor_x2, tensor_y2]], 'inplace_copy',
            disable_mode=['GRAPH_MODE', 'GRAPH_MODE_O0'])
    TEST_OP(copy_forward_func, [[tensor_x1, tensor_y1], [tensor_x2, tensor_y2]], 'inplace_copy', disable_grad=True,
            disable_mode=['GRAPH_MODE'])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_copy_bfloat16():
    """
    Feature: test copy functional API.
    Description: testcase for copy functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((3, 2, 5, 4), np.float32)
    y = generate_random_input((2, 1, 4), np.float32)

    expect = np.expand_dims(y.repeat(5, axis=1), axis=0).repeat(3, axis=0)
    expect_grad = np.ones_like(y).astype(np.float32) * 15

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output = copy_forward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))
    output_grad = copy_backward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))

    np.allclose(output.float().asnumpy(), expect, 0.004, 0.004, equal_nan=True)
    np.allclose(output_grad[1].float().asnumpy(), expect_grad, 0.004, 0.004, equal_nan=True)
