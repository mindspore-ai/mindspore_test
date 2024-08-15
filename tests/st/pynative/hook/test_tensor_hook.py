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
import os
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from tests.st.pynative.utils import GradOfAllParams, GradOfFirstInput
from tests.mark_utils import arg_mark

ms.set_context(mode=ms.PYNATIVE_MODE)


def hook_fn(grad_out):
    """改变梯度"""
    print("hook_fn print grad_out:", grad_out, flush=True)  # 该梯度是传播到该tensor时，该tensor所对应的梯度
    return grad_out * 2


def hook_fn_2(grad_out):
    """change gradient"""
    print("hook_fn print grad_out:", grad_out, flush=True)  # 该梯度是传播到该tensor时，该tensor所对应的梯度
    return grad_out * 3


def hook_test(x, y):
    z = x * y
    z.register_hook(hook_fn)  # 注册函数1
    z = z * y
    return z


def net(x, y):
    return ms.grad(hook_test, grad_position=(0, 1))(x, y)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_with_op_output():
    """
    Feature: Test tensor backward hook feature
    Description: test hook
    Expectation: Success
    """
    output = net(ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32))
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([2, 8, 18])).astype(np.float32).asnumpy(), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), Tensor(np.array([3, 12, 27])).astype(np.float32).asnumpy(), 0.001, 0.001)


def hook_test_multi(x, y):
    z = x * y
    z.register_hook(hook_fn)
    z.register_hook(hook_fn_2)
    z = z * y
    return z


def net_op_output_multi(x, y):
    return ms.grad(hook_test_multi, grad_position=(0, 1))(x, y)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_with_op_output_register_multi():
    """
    Feature: Test tensor backward hook feature
    Description: test hook
    Expectation: Success
    """
    output = net_op_output_multi(ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32),
                                 ms.Tensor(np.array([1.0, 2.0, 3.0]), ms.float32))
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([6, 24, 54])).astype(np.float32).asnumpy(), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), Tensor(np.array([7, 28, 63])).astype(np.float32).asnumpy(), 0.001, 0.001)


def hook_test_input(x):
    y1 = x ** 2
    y2 = x + 1
    return y1 + y2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_with_net_input():
    """
    Feature: Test tensor backward hook feature
    Description: test hook
    Expectation: Success
    """
    x = ms.Tensor(np.array([1.0]), ms.float32)
    x.register_hook(hook_fn)
    ms_grad = GradOfFirstInput(hook_test_input, False)
    output = ms_grad(x)
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([6])).astype(np.float32).asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_with_net_input_register_multi():
    """
    Feature: Test tensor backward hook feature
    Description: test hook
    Expectation: Success
    """
    x = ms.Tensor(np.array([1.0]), ms.float32)
    handle1 = x.register_hook(hook_fn)
    handle2 = x.register_hook(hook_fn_2)
    ms_grad = GradOfFirstInput(hook_test_input, False)
    output = ms_grad(x)
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([18])).astype(np.float32).asnumpy(), 0.001, 0.001)

    handle1.remove()
    ms_grad = GradOfFirstInput(hook_test_input, False)
    output = ms_grad(x)
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([9])).astype(np.float32).asnumpy(), 0.001, 0.001)

    handle1.remove()
    handle2.remove()
    ms_grad = GradOfFirstInput(hook_test_input, False)
    output = ms_grad(x)
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([3])).astype(np.float32).asnumpy(), 0.001, 0.001)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.weight1 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight1")
        self.weight2 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight2")
        self.handle1 = self.weight1.register_hook(hook_fn)
        self.handle2 = self.weight2.register_hook(hook_fn)

    def construct(self, x):
        y = x * self.weight1
        z = x * self.weight2
        return y + z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_with_weight():
    """
    Feature: Test tensor backward hook feature
    Description: test hook
    Expectation: Success
    """
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    net1 = Net()
    ms_grad = GradOfAllParams(net1, False)
    # First step
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([2, 4, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), Tensor(np.array([2, 4, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

    # Second step, no need register hook again
    input_x = Tensor(np.array([2.0, 3.0, 4.0]), ms.float32)
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([4, 6, 8])).astype(np.float32).asnumpy(), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), Tensor(np.array([4, 6, 8])).astype(np.float32).asnumpy(), 0.001, 0.001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_with_weight_register_multi():
    """
    Feature: Test tensor backward hook feature
    Description: test hook
    Expectation: Success
    """
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    net1 = Net()
    ms_grad = GradOfAllParams(net1, False)
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([2, 4, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), Tensor(np.array([2, 4, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

    # Add multi hook fn
    handle1 = net1.weight1.register_hook(hook_fn_2)
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([6, 12, 18])).astype(np.float32).asnumpy(), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), Tensor(np.array([2, 4, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

    # Remove original hook fn
    net1.handle1.remove()
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([3, 6, 9])).astype(np.float32).asnumpy(), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), Tensor(np.array([2, 4, 6])).astype(np.float32).asnumpy(), 0.001, 0.001)

    # remove all hook fn
    net1.handle1.remove()
    net1.handle2.remove()
    handle1.remove()
    output = ms_grad(input_x)
    assert np.allclose(output[0].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001, 0.001)
    assert np.allclose(output[1].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001, 0.001)


class NetRemove(nn.Cell):
    def __init__(self):
        super(NetRemove, self).__init__()
        self.weight1 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight1")
        self.handle = self.weight1.register_hook(hook_fn)

    def construct(self, x):
        x = x * self.weight1
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_backward_hook_handle_remove():
    """
    Feature: Test tensor backward hook feature
    Description: test hook
    Expectation: Success
    """
    input_x = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    net_remove = NetRemove()
    ms_grad = GradOfAllParams(net_remove, False)

    for i in range(2):
        if i == 0:
            output = ms_grad(input_x)
            assert np.allclose(output[0].asnumpy(), Tensor(np.array([2, 4, 6])).astype(np.float32).asnumpy(), 0.001,
                               0.001)
        else:
            net_remove.handle.remove()
            output = ms_grad(input_x)
            assert np.allclose(output[0].asnumpy(), Tensor(np.array([1, 2, 3])).astype(np.float32).asnumpy(), 0.001,
                               0.001)


@arg_mark(plat_marks=['platform_gpu'],
          level_mark='level0',
          card_mark='allcards',
          essential_mark='essential')
def test_tensor_hook_with_reduce_scatter():
    """
    Feature: mpi run 8P case of 'reduce_scatter' communication operator for pynative tensor hook.
    Description: mpi run 8P case of 'reduce_scatter' communication operator for pynative tensor hook.
    Expectation: success
    """
    return_code = os.system("mpirun --allow-run-as-root -n 8 pytest -s test_tensor_hook_reduce_scatter.py")
    assert return_code == 0
