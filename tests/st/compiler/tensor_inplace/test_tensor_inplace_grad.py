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
