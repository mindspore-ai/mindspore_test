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
from mindspore import Tensor, nn, context, Parameter
from mindspore.ops import operations as P

from tests.mark_utils import arg_mark
from tests.st.pynative.utils import GradOfAllInputs, GradOfAllParams


context.set_context(mode=ms.GRAPH_MODE)

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
        x.add_(y)
        return out


class AddInplaceParamNet1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.p = Parameter(Tensor([[2., 1.], [2., 3.]]), requires_grad=True)

    def construct(self, x, y):
        out = self.p * x + y
        x.add_(y)
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

    with pytest.raises(RuntimeError) as err:
        GradOfAllInputs(net, sens_param=True)(x, y, sens)
        torch_tensor_inplace_leaf_add(t_x, t_y)

    assert "a leaf Variable that requires grad \
        is being used in an in-place operation" in str(err.value)


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


@pytest.mark.skip(reason="Unsupported")
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


@pytest.mark.skip(reason="Unsupported")
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

    with pytest.raises(RuntimeError) as err:
        GradOfAllInputs(net, sens_param=True)(x, y, sens)
        torch_tensor_inplace_after_forward_add_backward_error(t_x, t_y)

    assert "one of the variables needed for gradient computation \
        has been modified by an inplace operation" in str(err.value)


@pytest.mark.skip(reason="Unsupported")
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


@pytest.mark.skip(reason="Unsupported")
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

    with pytest.raises(RuntimeError) as err:
        GradOfAllParams(net, sens_param=True)(x, y, sens)
        t_out.sum().backward()
    assert "one of the variables needed for gradient computation \
        has been modified by an inplace operation" in str(err.value)
