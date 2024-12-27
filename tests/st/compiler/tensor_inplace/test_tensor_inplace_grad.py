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
import torch
import mindspore as ms
from mindspore import Tensor, nn, context, Parameter, ParameterTuple
from mindspore import dtype as mstype
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.nn import Cell, BatchNorm2d, Conv2d, ParameterUpdate
from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs, GradOfAllParams
from tests.st.utils import test_utils


context.set_context(mode=ms.GRAPH_MODE)

class GradOfFirstInput(nn.Cell):
    def __init__(self, net):
        super(GradOfFirstInput, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)

class GradOfAllInputsAndParams(nn.Cell):
    def __init__(self, net):
        super(GradOfAllInputsAndParams, self).__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_all=True, get_by_list=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)

class LeafAddInplaceNet(nn.Cell):
    def construct(self, x, y):
        P.AssignAdd()(x, y)
        return x


class AddInplaceNet(nn.Cell):
    def construct(self, x, y):
        z = x + y
        P.AssignAdd()(z, x)
        out = z * x
        return out


class AddInplaceNet1(nn.Cell):
    def construct(self, x, y):
        z = x + y
        out = z + x
        P.AssignAdd()(z, x)
        return out


class AddInplaceNet2(nn.Cell):
    def construct(self, x, y):
        z = x + y
        out = z * x
        P.AssignAdd()(z, x)
        return out


class AddInplaceParamNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor([[2., 1.], [2., 3.]]), requires_grad=True)

    def construct(self, x, y):
        out = self.p * 2 + x + y
        P.AssignAdd()(x, y)
        return out


class AddInplaceParamNet1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor([[2., 1.], [2., 3.]]), requires_grad=True)

    def construct(self, x, y):
        out = self.p * x + y
        P.AssignAdd()(x, y)
        return out


class TorchAddInplaceParamNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.parameter.Parameter(
            torch.tensor([[2., 1.], [2., 3.]]), requires_grad=True)

    def forward(self, x, y):
        out = self.p * 2 + x + y
        x.add_(y)
        return out


class TorchAddInplaceParamNet1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.parameter.Parameter(
            torch.tensor([[2., 1.], [2., 3.]]), requires_grad=True)

    def forward(self, x, y):
        out = self.p * x + y
        x.add_(y)
        return out


def torch_tensor_inplace_leaf_add(x, y):
    x.add_(y)
    out = x.sum()
    return out

def torch_tensor_inplace_add_backward(x, y):
    z = x + y
    z.add_(x)
    out = z * x
    out.sum().backward()

def torch_tensor_inplace_after_forward_add_backward(x, y):
    z = x + y
    out = z + x
    z.add_(x)
    out.sum().backward()

def torch_tensor_inplace_after_forward_add_backward_error(x, y):
    z = x + y
    out = z * x
    z.add_(x)
    out.sum().backward()


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_leaf_add_grad():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    sens = Tensor([[1., 1.], [1., 1.]])
    net = LeafAddInplaceNet()
    x = Tensor([[3.0, 1.0], [2.0, 1.5]])
    y = Tensor([[2.0, 1.0], [2.0, 1.5]])

    t_x = torch.tensor([[3.0, 1.0], [2.0, 1.5]], requires_grad=True)
    t_y = torch.tensor([[2.0, 1.0], [2.0, 1.5]], requires_grad=True)

    with pytest.raises(RuntimeError) as err1:
        GradOfAllInputs(net, sens_param=True)(x, y, sens)
    assert "A leaf Variable that requires grad is being used in an in-place operation" in str(err1.value)

    with pytest.raises(RuntimeError) as err2:
        torch_tensor_inplace_leaf_add(t_x, t_y)
    assert "a leaf Variable that requires grad is being used in an in-place operation" in str(err2.value)


@pytest.mark.skip(reason="Unsupported")
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_grad():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    sens = Tensor([[1., 1.], [1., 1.]])
    net = AddInplaceNet()
    x = Tensor([[3.0, 1.0], [2.0, 1.5]])
    y = Tensor([[2.0, 1.5], [2.0, 1.5]])
    ms_grad = GradOfAllInputs(net, sens_param=True)(x, y, sens)

    t_x = torch.tensor([[3.0, 1.0], [2.0, 1.5]], requires_grad=True)
    t_y = torch.tensor([[2.0, 1.5], [2.0, 1.5]], requires_grad=True)
    torch_tensor_inplace_add_backward(t_x, t_y)
    np.testing.assert_almost_equal(ms_grad[0].asnumpy(), t_x.grad.numpy())
    np.testing.assert_almost_equal(ms_grad[1].asnumpy(), t_y.grad.numpy())


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_after_forward_add_grad():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    sens = Tensor([[1., 1.], [1., 1.]])
    net = AddInplaceNet1()
    x = Tensor([[3.0, 1.0], [2.0, 1.5]])
    y = Tensor([[2.0, 1.0], [2.0, 1.5]])
    ms_grad = GradOfAllInputs(net, sens_param=True)(x, y, sens)

    t_x = torch.tensor([[3.0, 1.0], [2.0, 1.5]], requires_grad=True)
    t_y = torch.tensor([[2.0, 1.0], [2.0, 1.5]], requires_grad=True)
    torch_tensor_inplace_after_forward_add_backward(t_x, t_y)

    np.testing.assert_almost_equal(ms_grad[0].asnumpy(), t_x.grad.numpy())
    np.testing.assert_almost_equal(ms_grad[1].asnumpy(), t_y.grad.numpy())


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_after_forward_add_grad_error():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    sens = Tensor([[1., 1.], [1., 1.]])
    net = AddInplaceNet2()
    x = Tensor([[3.0, 1.0], [2.0, 1.5]])
    y = Tensor([[3.0, 1.0], [2.0, 1.5]])

    t_x = torch.tensor([[3.0, 1.0], [2.0, 1.5]], requires_grad=True)
    t_y = torch.tensor([[3.0, 1.0], [2.0, 1.5]], requires_grad=True)

    with pytest.raises(RuntimeError) as err1:
        GradOfAllInputs(net, sens_param=True)(x, y, sens)
    assert ("One of the variables needed for gradient computation has been modified by an inplace operation"
            in str(err1.value))

    with pytest.raises(RuntimeError) as err2:
        torch_tensor_inplace_after_forward_add_backward_error(t_x, t_y)
    assert ("one of the variables needed for gradient computation has been modified by an inplace operation"
            in str(err2.value))


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_after_forward_add_param_grad():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    sens = Tensor([[1., 1.], [1., 1.]])
    net = AddInplaceParamNet()
    x = Tensor([[3.0, 1.0], [2.0, 1.5]])
    y = Tensor([[3.0, 1.0], [2.0, 1.5]])
    ms_param_grad = GradOfAllParams(net, sens_param=True)(x, y, sens)

    t_x = torch.tensor([[3.0, 1.0], [2.0, 1.5]])
    t_y = torch.tensor([[3.0, 1.0], [2.0, 1.5]])
    t_net = TorchAddInplaceParamNet()
    t_out = t_net(t_x, t_y)
    t_out.sum().backward()

    np.testing.assert_almost_equal(ms_param_grad[0].asnumpy(), t_net.p.grad.numpy())


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_after_forward_add_param_grad_error():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    sens = Tensor([[1., 1.], [1., 1.]])
    net = AddInplaceParamNet1()
    x = Tensor([[3.0, 1.0], [2.0, 1.5]])
    y = Tensor([[3.0, 1.0], [2.0, 1.5]])

    t_x = torch.tensor([[3.0, 1.0], [2.0, 1.5]])
    t_y = torch.tensor([[3.0, 1.0], [2.0, 1.5]])
    t_net = TorchAddInplaceParamNet1()
    t_out = t_net(t_x, t_y)

    with pytest.raises(RuntimeError) as err1:
        GradOfAllParams(net, sens_param=True)(x, y, sens)
    assert ("One of the variables needed for gradient computation has been modified by an inplace operation"
            in str(err1.value))

    with pytest.raises(RuntimeError) as err2:
        t_out.sum().backward()
    assert ("one of the variables needed for gradient computation has been modified by an inplace operation"
            in str(err2.value))


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_grad_error():
    """
    Feature: Support tensor inplace in grad.
    Description: Assert RuntimeError with correct error message and error line.
    Expectation: Run success.
    """
    class MixControlNet(Cell):
        def __init__(self, in_channel, x):
            super().__init__()
            self.biasadd = P.BiasAdd()
            self.addn = P.AddN()
            self.conv = Conv2d(in_channels=in_channel, out_channels=in_channel,
                               kernel_size=1, stride=1, has_bias=False,
                               weight_init='ones', pad_mode='same')
            self.bn = BatchNorm2d(num_features=in_channel)
            self.mean = P.ReduceMean(keep_dims=False)
            self.bias = Parameter(Tensor(np.random.randint(2, size=(3,)).astype((np.float32))),
                                  name="bias")
            self.bias2 = Parameter(Tensor(np.ones([3,]).astype(np.float32)),
                                   name="bias2")
            self.parameterupdate = ParameterUpdate(self.bias)
            self.x = x

        def construct(self, input_x):
            x = self.x
            z = self.x
            out = self.biasadd(input_x, self.bias)
            while x < 20:
                update = self.parameterupdate(self.bias2)
                out = self.biasadd(out, update)
                if x < 10:
                    out = self.addn((input_x, out))
                    while z < 20:
                        out = self.conv(out)
                        z = z + 1
                if x < 20:
                    out = self.biasadd(out, self.bias)
                    if x % 2 == 0:
                        out = self.biasadd(out, self.bias)
                        out = self.bn(out)
                    else:
                        out = self.conv(out)
                x = x + 1
            out = self.addn((out, out))
            out = self.mean(out, (2, 3))
            return out

    net = MixControlNet(3, 5)
    input_x = Tensor(np.random.randint(2, size=(1, 3, 2, 2)).astype((np.float32)))
    label = Tensor(np.zeros([1, 3]).astype(np.float32))
    with pytest.raises(RuntimeError) as info:
        opt = nn.Momentum(learning_rate=0.0001, momentum=0.009, params=net.trainable_params())
        loss = nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='mean')
        train_network = ms.amp.build_train_network(net, opt, loss, level="auto")
        train_network(input_x, label)
    assert "A leaf Variable that requires grad is being used in an in-place operation." in str(info.value)
    assert "update = self.parameterupdate(self.bias2)" in str(info.value)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_grad_first_input():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    class Net1(nn.Cell):
        def construct(self, x, y):
            P.AssignAdd()(y, x)
            return y

    x = Tensor([1], dtype=mstype.float32)
    y = Tensor([2], dtype=mstype.float32)
    output = GradOfFirstInput(Net1())(x, y)
    print("output:", output)
    assert output == 0


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_grad_first_input_wrong():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    class Net2(nn.Cell):
        def construct(self, x, y):
            P.AssignAdd()(x, y)
            return x

    with pytest.raises(RuntimeError) as info:
        x = Tensor([1], dtype=mstype.float32)
        y = Tensor([2], dtype=mstype.float32)
        output = GradOfFirstInput(Net2())(x, y)
        print("output:", output)
    assert "A leaf Variable that requires grad is being used in an in-place operation." in str(info.value)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_grad_all_inputs_and_param():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    class Net7(nn.Cell):
        def __init__(self):
            super(Net7, self).__init__()
            self.param1 = Parameter(Tensor([1], dtype=mstype.float32), name="param1")
            self.param2 = Parameter(Tensor([1], dtype=mstype.float32), name="param2")

        def construct(self, x, y):
            out = self.param1 + self.param2 + x + y
            out = out * x
            P.AssignAdd()(out, y)
            return out

    x = Tensor([1], dtype=mstype.float32)
    y = Tensor([2], dtype=mstype.float32)
    output = GradOfAllInputsAndParams(Net7())(x, y)
    print("output:", output)
    assert output == ((6, 1), (1, 1))


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_grad_all_inputs_and_param_wrong():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    class Net8(nn.Cell):
        def __init__(self):
            super(Net8, self).__init__()
            self.param1 = Parameter(Tensor([1], dtype=mstype.float32), name="param1")
            self.param2 = Parameter(Tensor([1], dtype=mstype.float32), name="param2")

        def construct(self, x, y):
            out = self.param1 + self.param2 + x + y
            out1 = out * x
            P.AssignAdd()(out, y)
            return out1

    with pytest.raises(RuntimeError) as info:
        x = Tensor([1], dtype=mstype.float32)
        y = Tensor([2], dtype=mstype.float32)
        output = GradOfAllInputsAndParams(Net8())(x, y)
        print("output:", output)
    assert ("One of the variables needed for gradient computation has been modified by an inplace operation."
            in str(info.value))


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_control_flow_grad():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            if x * 2 > y:
                z = x + y * x
                P.AssignAdd()(y, x)
            else:
                z = x - y
            return z
    with pytest.raises(RuntimeError) as info:
        input_x = ms.Tensor(2, dtype=ms.int32)
        input_y = ms.Tensor(3, dtype=ms.int32)
        out = GradOfAllInputsAndParams(Net())(input_x, input_y)
        print("out:", out)
    assert ("One of the variables needed for gradient computation has been modified by an inplace operation."
            in str(info.value))


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_control_flow_grad_2():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            if x * 2 > y:
                w = y + x
                z = w * x
            else:
                z = x - y
            P.AssignAdd()(y, x)
            return z

    with pytest.raises(RuntimeError) as info:
        input_x = ms.Tensor(2, dtype=ms.int64)
        input_y = ms.Tensor(3, dtype=ms.int64)
        output = GradOfFirstInput(Net())(input_x, input_y)
        print("output:", output)
    assert ("One of the variables needed for gradient computation has been modified by an inplace operation."
            in str(info.value))


@pytest.mark.skip(reason="Unsupported")
@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_control_flow_grad_param():
    """
    Feature: Support tensor inplace in grad.
    Description: Support tensor inplace in grad.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor(1, dtype=ms.int64), name='param')

        def construct(self, x, y):
            if x * 2 > y:
                w = self.param + x
                z = w * x
            else:
                w = y * 2
                z = x - y
            P.AssignAdd()(w, x)
            return z

    with pytest.raises(RuntimeError) as info:
        input_x = ms.Tensor(2, dtype=ms.int64)
        input_y = ms.Tensor(3, dtype=ms.int64)
        output = GradOfFirstInput(Net())(input_x, input_y)
        print("output:", output)
    assert ("One of the variables needed for gradient computation has been modified by an inplace operation."
            in str(info.value))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
def test_tensor_inplace_scatter_grad():
    """
    Feature: Support tensor scatter_ method in grad.
    Description: Support tensor scatter_ method in grad.
    Expectation: Run success.
    """
    class ScatterGrad(nn.Cell):
        def __init__(self, net: nn.Cell, sens: Tensor):
            super().__init__()
            self.net = net
            self.grad_op = ops.GradOperation(get_all=True, sens_param=True)
            self.grad_wrt_output = sens

        def construct(self, x, dim, index, src_or_val, reduce):
            return self.grad_op(self.net)(x, dim, index, src_or_val, reduce, self.grad_wrt_output)

    @test_utils.run_with_cell
    def scatter_val_with_grad(x, dim, index, value, reduce):
        return (x * True).scatter_(dim=dim, index=index, value=value,
                                   **(dict(reduce=reduce) if reduce != 'none' else {}))
    ## inplace backward
    context.set_context(jit_level='O0')
    slf = Tensor([[2] * 4] * 3, dtype=ms.float32)
    value = np.random.rand() * 10
    index = Tensor(np.array([list(range(3)) + [2]] * 3, dtype=np.int64))  # slf[:, 3] is reserved
    grad = Tensor(np.random.rand(3, 4), dtype=ms.float32)
    grad_np = grad.asnumpy().copy().astype(np.float32)
    grads = ScatterGrad(scatter_val_with_grad, grad)(slf, 1, index, value, 'none')
    # only self has grad
    grad_np[:, :3] = 0
    assert np.allclose(grads[0].asnumpy().astype(np.float32), grad_np)
