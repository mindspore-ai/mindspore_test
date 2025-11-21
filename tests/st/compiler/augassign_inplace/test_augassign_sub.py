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

"""test augassign sub."""

import pytest
from mindspore import nn
from mindspore import context, Tensor, jit, ParameterTuple
from mindspore import ops
from tests.mark_utils import arg_mark


class GradOfFirstInput(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = ops.GradOperation()

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)


class GradOfAllInputs(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net)
        return gradient_function(x, y)


class GradOfAllInputsAndParams(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.params = ParameterTuple(net.trainable_params())
        self.grad_op = ops.GradOperation(get_all=True, get_by_list=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net, self.params)
        return gradient_function(x, y)


class Net(nn.Cell):
    def construct(self, x, y):
        x -= y
        return x


def set_context_mode(input_net, mode):
    if mode == 'kbk':
        input_net.construct = jit(input_net.construct, backend='ms_backend')
    elif mode == 'ge':
        input_net.construct = jit(input_net.construct, backend='GE')
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['kbk', 'ge'])
def test_augassign_sub_tensor_scalar(mode):
    """
    Feature: Support augassign inplace.
    Description: Support augassign inplace.
    Expectation: Run success.
    """
    net = Net()
    x = Tensor(3)
    y = 2
    set_context_mode(net, mode)
    graph_ret = net(x, y)

    x2 = Tensor(3)
    y2 = 2
    set_context_mode(net, 'pynative')
    pynative_ret = net(x2, y2)
    assert graph_ret == pynative_ret


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['kbk', 'ge'])
def test_augassign_sub_tensor_tensor(mode):
    """
    Feature: Support augassign inplace.
    Description: Support augassign inplace.
    Expectation: Run success.
    """
    net = Net()
    x = Tensor(3)
    y = Tensor(1)
    set_context_mode(net, mode)
    graph_ret = net(x, y)

    x2 = Tensor(3)
    y2 = Tensor(1)
    set_context_mode(net, 'pynative')
    pynative_ret = net(x2, y2)
    assert graph_ret == pynative_ret


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", ['kbk', 'ge'])
def test_tensor_tensor_grad_first_input(mode):
    """
    Feature: Support augassign inplace in grad.
    Description: Support augassign inplace in grad.
    Expectation: Run success.
    """
    net = Net()
    x = Tensor(10)
    y = Tensor(3)
    set_context_mode(net, mode)
    graph_ret = GradOfFirstInput(net)(x, y)

    x2 = Tensor(10)
    y2 = Tensor(3)
    set_context_mode(net, 'pynative')
    pynative_ret = GradOfFirstInput(net)(x2, y2)
    assert graph_ret == pynative_ret


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", ['kbk', 'ge'])
def test_tensor_scalar_grad_first_input(mode):
    """
    Feature: Support augassign inplace in grad.
    Description: Support augassign inplace in grad.
    Expectation: Run success.
    """
    net = Net()
    x = Tensor(10)
    y = 3
    set_context_mode(net, mode)
    graph_ret = GradOfFirstInput(net)(x, y)

    x2 = Tensor(10)
    y2 = 3
    set_context_mode(net, 'pynative')
    pynative_ret = GradOfFirstInput(net)(x2, y2)
    assert graph_ret == pynative_ret


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", ['kbk', 'ge'])
def test_tensor_tensor_grad_all_inputs(mode):
    """
    Feature: Support augassign inplace in grad.
    Description: Support augassign inplace in grad.
    Expectation: Run success.
    """
    net = Net()
    x = Tensor(10)
    y = Tensor(3)
    set_context_mode(net, mode)
    graph_ret = GradOfAllInputs(net)(x, y)

    x2 = Tensor(10)
    y2 = Tensor(3)
    set_context_mode(net, 'pynative')
    pynative_ret = GradOfAllInputs(net)(x2, y2)
    assert graph_ret == pynative_ret


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", ['kbk', 'ge'])
def test_tensor_scalar_grad_all_inputs(mode):
    """
    Feature: Support augassign inplace in grad.
    Description: Support augassign inplace in grad.
    Expectation: Run success.
    """
    net = Net()
    x = Tensor(10)
    y = 3
    set_context_mode(net, mode)
    graph_ret = GradOfAllInputs(net)(x, y)

    x2 = Tensor(10)
    y2 = 3
    set_context_mode(net, 'pynative')
    pynative_ret = GradOfAllInputs(net)(x2, y2)
    assert graph_ret == pynative_ret


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['kbk', 'ge'])
def test_tensor_tensor_grad_all_inputs_and_params(mode):
    """
    Feature: Support augassign inplace in grad.
    Description: Support augassign inplace in grad.
    Expectation: Run success.
    """
    net = Net()
    x = Tensor(10)
    y = Tensor(3)
    set_context_mode(net, mode)
    graph_ret = GradOfAllInputsAndParams(net)(x, y)

    x2 = Tensor(10)
    y2 = Tensor(3)
    set_context_mode(net, 'pynative')
    pynative_ret = GradOfAllInputsAndParams(net)(x2, y2)
    assert graph_ret == pynative_ret


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("mode", ['kbk', 'ge'])
def test_tensor_scalar_grad_all_inputs_and_params(mode):
    """
    Feature: Support augassign inplace in grad.
    Description: Support augassign inplace in grad.
    Expectation: Run success.
    """
    net = Net()
    x = Tensor(10)
    y = 3
    set_context_mode(net, mode)
    graph_ret = GradOfAllInputsAndParams(net)(x, y)

    x2 = Tensor(10)
    y2 = 3
    set_context_mode(net, 'pynative')
    pynative_ret = GradOfAllInputsAndParams(net)(x2, y2)
    assert graph_ret == pynative_ret
