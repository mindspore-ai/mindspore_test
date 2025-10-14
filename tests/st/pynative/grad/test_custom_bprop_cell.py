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
# ============================================================================
""" test_bprop """
import numpy as np
import pytest
import mindspore as ms
import mindspore.nn as nn
from mindspore.common import Tensor
from mindspore.ops import composite as C
from tests.mark_utils import arg_mark


class CustomBpropNet(nn.Cell):
    def construct(self, x):
        y = x * x
        z = y + y
        return z

    def bprop(self, *args):
        return (args[0] * 4,)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_bprop_net():
    """
    Feature: Test auto grad stop gradient.
    Description: Test auto grad stop gradient.
    Expectation: Success.
    """
    x = Tensor([2], ms.float32)
    net = CustomBpropNet()
    grad = ms.grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([8], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
class NoneCustomNet(nn.Cell):
    def construct(self, x, y):
        y = x * x
        return y

    def bprop(self, *args):
        return args[0] * 2, None


class NoneAddNet(nn.Cell):
    def __init__(self):
        super(NoneAddNet, self).__init__()
        self.net = NoneCustomNet()

    def construct(self, x):
        y = x * x
        output = self.net(x, y)
        h = y + output
        return h


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_auto_grad_none_add_net():
    """
    Feature: Test auto grad none add
    Description: Test auto grad none add.
    Expectation: Success.
    """
    x = Tensor([2.0], ms.float32)
    net = NoneAddNet()
    grad = ms.grad(net)(x)
    assert np.allclose(grad.asnumpy(), np.array([8.], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionAutoReduceNet(nn.Cell):
    def construct(self, x, y):
        x2 = x + y
        return x2

    def bprop(self, *args):
        return Tensor([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]]), Tensor([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]])


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_auto_reduce():
    """
    Feature: Custom bprop function.
    Description: Test auto reduce.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], ms.float32)
    net = CustomFunctionAutoReduceNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([4., 4., 4.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]], dtype=np.float32),
                       0.00001, 0.00001)


class CustomFunctionAutoCastNet(nn.Cell):
    def construct(self, x, y):
        x2 = x + y
        return x2

    def bprop(self, *args):
        return Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=ms.int64), Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]],
                                                                                 dtype=ms.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_auto_cast():
    """
    Feature: Custom bprop function.
    Description: Test auto cast.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], ms.float32)
    net = CustomFunctionAutoCastNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert grads[0].dtype == ms.float32
    assert grads[1].dtype == ms.float32
    assert np.allclose(grads[0].asnumpy(), np.array([4., 4., 4.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([[1., 1., 1.], [1., 1., 1.], [2., 2., 2.]], dtype=np.float32),
                       0.00001, 0.00001)


class CustomFunctionBroadcastExecptionNet(nn.Cell):
    def construct(self, x, y):
        x2 = x + y
        return x2

    def bprop(self, *args):
        return Tensor([[1, 1, 1, 1], [1, 1, 1, 1], [2, 2, 2, 2]], dtype=ms.int64), \
               Tensor([[1, 1, 1], [1, 1, 1], [2, 2, 2]], dtype=ms.int64)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_reduce_exception():
    """
    Feature: Custom bprop function.
    Description: Test auto reduce.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    y = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], ms.float32)
    net = CustomFunctionBroadcastExecptionNet()
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "For custom function, grad tensor should be broadcast to" in str(err.value)


class CustomFunctionReturnSelfNet(nn.Cell):
    def construct(self, x):
        return x

    def bprop(self, *args):
        return Tensor([1, 1, 1], dtype=ms.float32)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_return_self_net():
    """
    Feature: Custom bprop function.
    Description: Test bprop function return self.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    net = CustomFunctionReturnSelfNet()
    net.set_grad()
    output = net(x)
    grad_net = C.GradOperation(get_all=True)
    grad_net(net)(x)
    assert id(output) != id(x)


class CustomFunctionMultiOutputReturnSelfNet(nn.Cell):
    def construct(self, x):
        return x, Tensor([3, 3, 3], ms.float32)

    def bprop(self, *args):
        return Tensor([1, 1, 1], dtype=ms.float32)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_output_return_self_net():
    """
    Feature: Custom bprop function.
    Description: Test bprop function return self.
    Expectation: success.
    """
    x = Tensor([3, 3, 3], ms.float32)
    net = CustomFunctionMultiOutputReturnSelfNet()
    net.set_grad()
    output = net(x)
    grad_net = C.GradOperation(get_all=True)
    grad_net(net)(x)
    assert id(output[0]) != id(x)
