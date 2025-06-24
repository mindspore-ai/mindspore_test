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
import os
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
from mindspore.ops.auto_generate.gen_ops_prim import select_ext_view_op, slice_ext_view_op, inplace_copy_op
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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
    out_expect = grad(net, grad_position=(0, 1))(Tensor(3), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor(3), Tensor([[1, 2], [3, 4]]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


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
    out_expect = grad(net, grad_position=(0, 1))(Tensor(3), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor(3), Tensor([[1, 2], [3, 4]]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


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
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 1]), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 1]), Tensor([[1, 2], [3, 4]]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_check1():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1):
            input_tensor1_1 = ops.abs(input_tensor1)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(x, 0, 1)
            y.add_(2)
            z = x * 2
            z.add_(3)
            return z

    net = Net()
    out_expect = grad(net)(Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([[1, 2], [3, 4]]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_check2():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1):
            input_tensor1_1 = ops.abs(input_tensor1)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            x.add_(4)
            z = x * 2
            z.add_(3)
            return z

    net = Net()
    out_expect = grad(net)(Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_check3():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor2_1, 0, 0)
            x.add_(3)
            y.add_(2)
            z = input_tensor1_1 * 2
            z.add_(3)
            return z

    net = Net()
    out_expect = grad(net)(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_check5():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor2_1, 0, 0)
            y.add_(2)
            x.add_(6)
            return x, y

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_check6():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def inner_func(self, x, y):
            y.add_(2)
            z = x * 2
            z.add_(3)
            return z

        def func(self, input_tensor1_1, input_tensor2_1):
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor2_1, 0, 0)
            if x < self.inner_func(x, y):
                return y
            return x

        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            return self.func(input_tensor1_1, input_tensor2_1)

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_check7():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def construct(self, input_tensor2, input_tensor1):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            x.add_(3)
            z = input_tensor1_1 * 2
            z.add_(3)
            input_tensor2_1.add_(2)
            return input_tensor1_1 + z

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_check8():
    """
    Feature: view inplace operation in grad.
    Description: view inplace operation in grad.
    Expectation: no exception
    """

    class Net(nn.Cell):
        def func(self, input_tensor1):
            x = select_ext_view_op(input_tensor1, 0, 0)
            x.add_(3)
            z = x * 2
            z.add_(3)
            return z

        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            y = select_ext_view_op(input_tensor1_1, 0, 0)
            y.add_(2)
            z = self.func(input_tensor1_1)
            if (z > 3).all():
                return input_tensor1_1 + 1
            return z + 1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 2]), Tensor([3, 4]))
    assert np.allclose(out_expect[0].asnumpy(), out_jit[0].asnumpy())
    assert np.allclose(out_expect[1].asnumpy(), out_jit[1].asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_setitem_simple_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        @ms.jit(jit_level="O0", backend="ms_backend")
        def construct(self, x, a):
            x[0] = a
            y = x[1][1] # pylint: disable=unused-variable
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        x = Tensor([[2, 2, 2], [3, 3, 3]])
        a = Tensor([1, 1, 1])
        grad_net = ops.grad(net, grad_position=0)
        x_grad = grad_net(x, a)
        assert np.all(x_grad.asnumpy() == Tensor([[0, 0, 0], [1, 1, 1]]).asnumpy())

        grad_net = ops.grad(net, grad_position=1)
        a_grad = grad_net(x, a)
        assert np.all(a_grad.asnumpy() == Tensor([1, 1, 1]).asnumpy())
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_setitem_simple_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        @ms.jit(jit_level="O0", backend="ms_backend")
        def construct(self, x, a):
            y = x[1]
            y[0] = a
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        x = Tensor([[2, 2, 2], [3, 3, 3]])
        a = Tensor([3])
        grad_net = ops.grad(net, grad_position=0)
        x_grad = grad_net(x, a)
        assert np.all(x_grad.asnumpy() == Tensor([[1, 1, 1], [0, 1, 1]]).asnumpy())

        grad_net = ops.grad(net, grad_position=1)
        a_grad = grad_net(x, a)
        assert np.all(a_grad.asnumpy() == Tensor([1]).asnumpy())
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_setitem_simple_case3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        @ms.jit(jit_level="O0", backend="ms_backend")
        def construct(self, x, a):
            y = x[1]
            y[0] = a
            return y
    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        x = Tensor([[2, 2, 2], [3, 3, 3]])
        a = Tensor([3])
        grad_net = ops.grad(net, grad_position=0)
        x_grad = grad_net(x, a)
        assert np.all(x_grad.asnumpy() == Tensor([[0, 0, 0], [0, 1, 1]]).asnumpy())

        grad_net = ops.grad(net, grad_position=1)
        a_grad = grad_net(x, a)
        assert np.all(a_grad.asnumpy() == Tensor([1]).asnumpy())
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]
