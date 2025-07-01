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
import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
from mindspore.ops.auto_generate.gen_ops_prim import select_ext_view_op, InplaceMul
from mindspore.ops.auto_generate.gen_ops_def import inplace_add_ext_op
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_classic_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            inplace_add_ext_op(y_viewed, value)
            return y

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[0, 1], [2, 3]], dtype=ms.float32),
                                                 Tensor([1, 2], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[0, 1], [2, 3]], dtype=ms.float32),
                                              Tensor([1, 2], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_classic_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            InplaceMul()(y_viewed, value)
            return y

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


# scene1:
# multi node with same area
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene1_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            m.add_(x)
            n = select_ext_view_op(input_tensor1, 0, 0)
            n.mul_(2)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 2], dtype=ms.float32),
                                                 Tensor([[1, 2], [3, 4]], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 2], dtype=ms.float32),
                                              Tensor([[1, 2], [3, 4]], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene2_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            y_viewed2 = select_ext_view_op(y, 0, 0)
            inplace_add_ext_op(y_viewed2, y_viewed)
            return y

    net = Net()
    out_expect = grad(net)(Tensor([[1, 2], [3, 4]], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([[1, 2], [3, 4]], dtype=ms.float32))
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


# scene2:
# x[0] += x[1]
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene2_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            m.add_(x)
            n = select_ext_view_op(input_tensor1, 0, 1)
            n.add_(m)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([1, 2], dtype=ms.float32),
                                                 Tensor([[1, 2], [3, 4]], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([1, 2], dtype=ms.float32),
                                              Tensor([[1, 2], [3, 4]], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


# scene3:
# nested view
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene3_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            y_viewed2 = select_ext_view_op(y_viewed, 0, 0)
            InplaceMul()(y_viewed2, value)
            return y

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene3_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, input_tensor, x):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            m.add_(x)
            n = select_ext_view_op(m, 0, 1)
            n.add_(n)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene3_case3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, input_tensor, x):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            m = m.add_(x)
            n = select_ext_view_op(m, 0, 1)
            n.add_(n)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


# scene4:
# return viewed_output
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene4_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, input_tensor, x):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            m.add_(x)
            n = select_ext_view_op(input_tensor1, 0, 1)
            n.add_(x)
            return n

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_virtual_view_classic_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            InplaceMul()(m, value)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="The gradient error for 'value' is unrelated to virtual view insertion, fix it later")
def test_virtual_view_classic_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            InplaceMul()(m, value)
            InplaceMul()(m, value)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_virtual_view_classic_case3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            InplaceMul()(y, value)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


# scene5:
# inplace op change x_view_input before view op
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene5_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, input_tensor, x):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            input_tensor1.mul_(2)
            m.add_(x)
            return input_tensor1

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene5_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, input_tensor, x):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            input_tensor1.mul_(2)
            m.add_(x)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


# scene6:
# view_output1 and view_output2
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene6_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, input_tensor, x):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            m.add_(x)
            n = select_ext_view_op(input_tensor1, 1, 0)
            n.add_(x)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


# scene7:
# nested view return view
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene7_case1():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            m.add_(value)
            n = select_ext_view_op(m, 0, 1)
            n.add_(n)
            return n

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene7_case2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            m.add_(value)
            n = select_ext_view_op(m, 0, 1)
            n.add_(n)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene7_case3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            n = select_ext_view_op(m, 0, 1)
            InplaceMul()(m, value)
            return n

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene7_case4():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            n = select_ext_view_op(m, 0, 1)
            m.add_(value)
            n.add_(n)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene7_case5():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            n = select_ext_view_op(m, 0, 1)
            n.add_(n)
            z = m.add(m)
            z.add_(m)
            return n

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene7_case6():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            n = select_ext_view_op(m, 0, 1)
            y.add_(2)
            m.add_(value)
            y.add_(2)
            n.add_(n)
            return m

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor([6, 7], dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor([6, 7], dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_scene7_case7():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, value):
            y = ops.abs(x)
            m = select_ext_view_op(y, 0, 0)
            n = select_ext_view_op(m, 0, 1)
            u = select_ext_view_op(y, 0, 0)
            v = select_ext_view_op(u, 0, 1)
            InplaceMul()(n, value)
            return v

    net = Net()
    out_expect = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                                 Tensor(2, dtype=ms.float32))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net, grad_position=(0, 1))(Tensor([[1, 2], [3, 4]], dtype=ms.float32),
                                              Tensor(2, dtype=ms.float32))
    assert (out_expect[0].asnumpy() == out_jit[0].asnumpy()).all()
    assert (out_expect[1].asnumpy() == out_jit[1].asnumpy()).all()
