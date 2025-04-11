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
    net = Net()
    out_expect = grad(net)(x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


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
    net = Net()
    out_expect = grad(net)(x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


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
    net = Net()
    out_expect = grad(net)(x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


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
    net = Net()
    out_expect = grad(net)(x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


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

    net = Net()
    out_expect = grad(net)(Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([3, 4]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@pytest.mark.skip(reason="No support")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_view_and_inplace_grad_change_same_area1():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            t = select_ext_view_op(m, 0, 0)
            t.add_(x)
            n = select_ext_view_op(input_tensor1, 0, 0)
            z = select_ext_view_op(n, 0, 0)
            z.mul_(2)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    assert out_expect == out_jit


@pytest.mark.skip(reason="No support")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_view_and_inplace_grad_change_same_area2():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            t = select_ext_view_op(m, 0, 0)
            t.add_(x)
            input_tensor1.add_(input_tensor1)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    assert out_expect == out_jit


@pytest.mark.skip(reason="No support")
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_view_and_inplace_grad_change_same_area3():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            m.add_(x)
            n = select_ext_view_op(m, 0, 0)
            n.add_(n)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor([1, 1]), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 1]), Tensor([[1, 2], [3, 4]]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())
