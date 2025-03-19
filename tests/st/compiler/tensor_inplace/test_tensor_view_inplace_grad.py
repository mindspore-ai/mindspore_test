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
# ==============================================================================
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
from mindspore.ops.auto_generate.gen_ops_prim import select_ext_view_op, slice_ext_view_op, inplace_copy_op
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

class GradNet(nn.Cell):
    def __init__(self, net):
        super(GradNet, self).__init__()
        self.grad_op = ops.grad(net)

    def construct(self, x):
        return self.grad_op(x)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_once():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            inplace_copy_op(y_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y

    x = ms.Tensor([[0, 1], [2, 3]], dtype=ms.float32)
    forward_net = Net()
    grad_res = GradNet(forward_net)(x)
    expected_res = np.array([[0, 0], [1, 1]]).astype(np.float32)
    assert (grad_res.asnumpy() == expected_res).all()



@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_twice():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed = slice_ext_view_op(y, 1, 1, 2, 1)
            z_viewed = slice_ext_view_op(y_viewed, 0, 0, 1, 1)
            inplace_copy_op(z_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y

    x_np = (np.arange(2 * 2 * 2)).reshape((2, 2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    forward_net = Net()
    grad_res = GradNet(forward_net)(x)
    expected_res = np.array([[[0, 1], [0, 0]], [[1, 1], [1, 1]]]).astype(np.float32)
    assert (grad_res.asnumpy() == expected_res).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_grad():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed1 = slice_ext_view_op(y, 0, 0, 1, 1)
            z = y_viewed1 + 1
            y_viewed2 = slice_ext_view_op(y, 0, 0, 1, 1)
            inplace_copy_op(y_viewed2, z)
            return y

    x_np = (np.arange(2 * 2)).reshape((2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    forward_net = Net()
    grad_res = GradNet(forward_net)(x)
    expected_res = np.array([[0, 1], [1, 1]]).astype(np.float32)
    assert (grad_res.asnumpy() == expected_res).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_grad1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed1 = select_ext_view_op(y, 0, 0)
            inplace_copy_op(y_viewed1, ms.Tensor(-1, dtype=ms.float32))
            y_viewed2 = select_ext_view_op(y, 0, 1)
            inplace_copy_op(y_viewed2, ms.Tensor(-1, dtype=ms.float32))
            return y

    x_np = (np.arange(2 * 2)).reshape((2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    forward_net = Net()
    grad_res = GradNet(forward_net)(x)
    expected_res = np.array([[0, 0], [0, 0]]).astype(np.float32)
    assert (grad_res.asnumpy() == expected_res).all()


@pytest.mark.skip(reason="No support")
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1):
            input_tensor1_1 = ops.abs(input_tensor1)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor1_1, 0, 1)
            x.add_(y)
            return x

    ms.context.set_context(mode=ms.PYNATIVE_MODE, jit_level="O0")
    net = Net()
    out_expect = grad(net)(Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([3, 4]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())
