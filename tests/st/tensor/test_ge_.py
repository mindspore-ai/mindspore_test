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
import mindspore.nn as nn
from mindspore import Tensor
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


class Net(nn.Cell):
    def construct(self, x, y):
        return x.ge_(y)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential',
)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_tensor_ge__normal(mode):
    """
    Feature: tensor.ge_
    Description: Verify the result of tensor.ge_
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    net = Net()
    x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), ms.float32)
    y = Tensor(np.array([[1.5, -2.0], [2.5, 4.0]]), ms.float32)
    expect = np.array([[0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    output = net(x, y)
    np.testing.assert_allclose(expect, output.asnumpy(), rtol=1e-4, atol=1e-4)

    x = Tensor(np.array([1, -2, -4]), ms.int32)
    y = -2
    expect = np.array([1, 1, 0], dtype=np.int32)
    output = net(x, y)
    np.testing.assert_allclose(expect, output.asnumpy(), rtol=0, atol=0)

    x = Tensor(np.array([[True, False], [True, True]]), ms.bool_)
    y = Tensor(np.array([True, True]), ms.bool_)
    expect = np.array([[True, False], [True, True]], dtype=np.bool_)
    output = net(x, y)
    np.testing.assert_allclose(expect, output.asnumpy(), rtol=0, atol=0)


@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential',
)
def test_tensor_ge__dynamic():
    """
    Feature: tensor.ge_
    Description: test tensor.ge_ with dynamic shape
    Expectation: success
    """
    x1 = generate_random_input((2, 3), np.float32)
    y1 = generate_random_input((2, 3), np.float32)
    x2 = generate_random_input((4, 5, 6), np.float32)
    y2 = generate_random_input((4, 5, 6), np.float32)
    net = Net()
    TEST_OP(
        net,
        [[Tensor(x1), Tensor(y1)], [Tensor(x2), Tensor(y2)]],
        "inplace_greater_equal_tensor",
        disable_mode=["GRAPH_MODE"],
        disable_grad=True,
        inplace_update=True
    )

    x3 = generate_random_input((2, 2), np.float32)
    y3 = np.random.randn()
    x4 = generate_random_input((3, 4, 5), np.float32)
    y4 = np.random.randn()
    TEST_OP(
        net,
        [[Tensor(x3), y3], [Tensor(x4), y4]],
        "inplace_greater_equal_scalar",
        disable_mode=["GRAPH_MODE"],
        disable_grad=True,
        inplace_update=True
    )
