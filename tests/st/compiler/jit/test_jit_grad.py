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
import pytest
import numpy as np
from mindspore.common import Tensor, Parameter
from mindspore.common import dtype as mstype
from mindspore import context, jit, ops, nn
from mindspore.ops import composite as C
from mindspore.ops.composite import GradOperation
from mindspore.nn import Cell
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_tuple():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(x, y):
        return 2 * x[0] + y

    def func(x, y):
        x = x * 3
        return inner_func((x,), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_tuple_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(m):
        return 2 * m[0][0] + m[1]

    def func(x, y):
        x = x * 3
        return inner_func(((x,), y))

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_list():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(x, y):
        return 2 * x[0] + y

    def func(x, y):
        x = x * 3
        return inner_func([x,], y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_list_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(m):
        return 2 * m[0][0] + m[1]

    def func(x, y):
        x = x * 3
        return inner_func([[x,], y])

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_dict():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(m):
        return 2 * m["x"] + m["y"]

    def func(x, y):
        x = x * 3
        return inner_func({"x": x, "y": y})

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_dict_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(m):
        return 2 * m["x"][0] + m["y"]

    def func(x, y):
        x = x * 3
        return inner_func({"x": (x,), "y": y})

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_vargs():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(*args):
        return 2 * args[0][0] + args[1]

    def func(x, y):
        x = x * 3
        return inner_func((x,), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_vargs_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(*args):
        return 2 * args[0][0][0] + args[1]

    def func(x, y):
        x = x * 3
        return inner_func(((x,),), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@pytest.mark.skip(reason="Jit handle kwargs with mutable error, fix later")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_kwargs():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(**kwargs):
        return 2 * kwargs["m"][0] + kwargs["n"]

    def func(x, y):
        x = x * 3
        return inner_func(m=(x,), n=y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@pytest.mark.skip(reason="Jit handle kwargs with mutable error, fix later")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_kwargs_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(**kwargs):
        return 2 * kwargs["m"][0][0] + kwargs["n"]

    def func(x, y):
        x = x * 3
        return inner_func(m=([x,],), n=y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@pytest.mark.skip(reason="Jit handle kwargs with mutable error, fix later")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_kwargs_3():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(**kwargs):
        return 2 * kwargs["m"][0] + kwargs["n"]

    def func(x, y):
        return inner_func(m=mutable((x,)), n=y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([2, 2, 2]))


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pyboost_series_highgrad():
    """
    Feature: Test grad scene for highgrad with out jit
    Description: Test grad scene for highgrad with out jit
    Expectation: Success
    """
    class Net(nn.Cell):
        def __init__(self, num_layer):
            super().__init__()
            self.layers = nn.CellList()
            self.dense = nn.Dense(4, 4)
            for _ in range(num_layer):
                self.layers.append(nn.ReLU())
            self.flatten = nn.Flatten()

        def construct(self, x):
            out = x
            out = self.dense(x)
            for layer in self.layers:
                out = layer(out)
            out = self.flatten(out)
            return out

    class Grad(nn.Cell):
        def __init__(self, network):
            super(Grad, self).__init__()
            self.grad = C.GradOperation(get_all=True, sens_param=False)
            self.network = network

        def construct(self, x):
            gout = self.grad(self.network)(x)
            return gout

    net = Net(100)
    grad_net = Grad(net)
    d = Tensor(shape=[None, None], dtype=mstype.float32)
    grad_net.set_inputs(d)

    x = Tensor(np.random.randn(4, 4).astype(np.float32))
    ggrad_net = Grad(grad_net)
    grad = ggrad_net(x)
    assert np.all(grad[0].asnumpy() == 0)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_invalid_input():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: RuntimeError.
    """
    @jit
    def inner_func(m):
        return 2 * m["x"][0] + m["y"]

    def func(x, y):
        x = x * 3
        return inner_func({"x": (x, "a"), "y": y})

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])

    with pytest.raises(RuntimeError) as error_info:
        GradOperation()(func)(a, b)
    assert "contains tensor with gradient but can not mutable" in str(error_info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_invalid_input_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: RuntimeError.
    """
    @jit
    def inner_func(x, y):
        return 2 * x[0] + y

    def func(x, y):
        x = x * 3
        return inner_func((x, None), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])

    with pytest.raises(RuntimeError) as error_info:
        GradOperation()(func)(a, b)
    assert "contains tensor with gradient but can not mutable" in str(error_info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_dynamic_shape_change_param():
    """
    Feature: Test grad jit scene for dynamic shape change param.
    Description: Test grad jit scene for dynamic shape change param.
    Expectation: Success.
    """
    class Net_JIT(Cell):
        def __init__(self):
            super().__init__()
            self.num = 2

        @jit
        def construct(self, x, y):
            ops.assign_add(x, y)
            return y * y * self.num

    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.num = 2

        def construct(self, x, y):
            ops.assign_add(x, y)
            return y * y * self.num

    def compare_result(grad, grad_jit):
        for g1, g2 in zip(grad, grad_jit):
            if g1 is None:
                assert g2 is None
                continue
            assert np.allclose(g1.numpy(), g2.asnumpy(), 0.0001, 0.0001)

    net_jit = Net_JIT()
    grad_net_jit = GradOfAllInputs(net_jit, False)
    net = Net()
    grad_net = GradOfAllInputs(net, False)
    x1 = Tensor(np.random.rand(2, 3, 4), mstype.float32)
    x1 = Parameter(x1, name="x1")
    y1 = Tensor(np.random.rand(2, 3, 4), mstype.float32)
    grad1 = grad_net(x1, y1)
    grad1_jit = grad_net_jit(x1, y1)
    compare_result(grad1, grad1_jit)
    x2 = Tensor(np.random.rand(3, 3, 4), mstype.float32)
    x2 = Parameter(x2, name="x2")
    y2 = Tensor(np.random.rand(3, 3, 4), mstype.float32)
    grad2 = grad_net(x2, y2)
    grad2_jit = grad_net_jit(x2, y2)
    compare_result(grad2, grad2_jit)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_custom_bprop():
    """
    Feature: Custom cell bprop.
    Description: Test grad jit scene for custom cell bprop.
    Expectation: Success.
    """
    class SubNet(nn.Cell):
        def __init__(self):
            super(SubNet, self).__init__()
            self.matmul = ops.MatMul()

        def construct(self, x, y):
            out = self.matmul(x, y)
            return out

        def bprop(self, x, y, out, dout):
            return 2 * x, 2 * y

    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sub_net = SubNet()
            self.z = Parameter(Tensor(np.array([1.0], np.float32)), name='z')

        @jit
        def construct(self, x, y):
            x = x * self.z
            out = self.sub_net(x, y)
            return out

    class GradNetWrtX(nn.Cell):
        def __init__(self, net):
            super(GradNetWrtX, self).__init__()
            self.net = net
            self.grad_op = ops.GradOperation(get_all=True)

        def construct(self, x, y):
            gradient_function = self.grad_op(self.net)
            return gradient_function(x, y)

    x = Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32)
    y = Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)
    output = GradNetWrtX(Net())(x, y)
    expect_dx = np.array([[1.0, 1.2, 0.8],
                          [2.4, 2.6, 2.2]]).astype(np.float32)
    expect_dy = np.array([[0.02, 0.6, 2.2],
                          [0.2, 0.4, 2.6],
                          [4.2, 2.4, 6.6]]).astype(np.float32)
    assert np.allclose(output[0].asnumpy(), expect_dx)
    assert np.allclose(output[1].asnumpy(), expect_dy)
