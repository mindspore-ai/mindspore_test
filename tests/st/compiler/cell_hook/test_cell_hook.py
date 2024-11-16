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
from mindspore import Tensor, ops, nn, Parameter, ParameterTuple
from mindspore import dtype as mstype
from mindspore._extends.parse import compile_config
import mindspore.context as context
from tests.mark_utils import arg_mark


def double_gradient(gradient):
    return gradient * 2

def quard_gradient(gradient):
    return gradient * 4

def param_hook1(parameters):
    print("Enter param hook1")
    for param in parameters:
        print(param[0], param[1])
    print("Leave param hook1")


def param_hook2(parameters):
    print("Enter param hook2")
    for param in parameters:
        print(param[0], param[1])
    print("Leave param hook2")


def gradient_hook1(gradients):
    print("Enter gradient hook1")
    outs = []
    for gradient in gradients:
        outs.append(double_gradient(gradient[1]))
        print(gradient[0], gradient[1])
    print("Leave gradient hook1")
    return outs


def gradient_hook2(gradients):
    print("Enter gradient hook2")
    outs = []
    for gradient in gradients:
        outs.append(double_gradient(gradient[1]))
        print(gradient[0], gradient[1])
    print("Leave gradient hook2")
    return outs

def param_hook3(parameters):
    print("Enter param hook3")
    for param in parameters:
        print(param[0], param[1])
    print("Leave param hook3")

def gradient_hook3(gradients):
    print("Enter gradient hook3")
    outs = []
    for gradient in gradients:
        outs.append(quard_gradient(gradient[1]))
        print(gradient[0], gradient[1])
    print("Leave gradient hook3")
    return outs

class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.a1 = Parameter(Tensor(np.array([1.0], np.float32)), name='a1')
        self.a2 = Parameter(Tensor(np.array([2.0], np.float32)), name='a2')

    def construct(self, x):
        out = x * (self.a1 + self.a2)
        return out


class Net2(nn.Cell):
    def __init__(self, net1):
        super(Net2, self).__init__()
        self.net1 = net1
        self.matmul = ops.MatMul()
        self.b1 = Parameter(Tensor(np.array([3.0], np.float32)), name='b1')
        self.b2 = Parameter(Tensor(np.array([4.0], np.float32)), name='b2')

    def construct(self, x, y):
        x = x * self.net1(x)
        x = x * self.b1
        x = x * self.b2
        out = self.matmul(x, y)
        return out


class GradNet(nn.Cell):
    def __init__(self, net2, get_all=False, get_by_list=False):
        super(GradNet, self).__init__()
        self.get_by_list = get_by_list
        self.net2 = net2
        self.c1 = Parameter(Tensor(np.array([5.0], np.float32)), name='c1')
        self.params = ParameterTuple(net2.trainable_params())
        self.grad_op = ops.GradOperation(get_all=get_all, get_by_list=get_by_list)

    def construct(self, x, y):
        print(self.c1)
        if self.get_by_list:
            gradient_function = self.grad_op(self.net2, self.params)
        else:
            gradient_function = self.grad_op(self.net2)
        grad = gradient_function(x, y)
        return grad


input_x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
input_y = Tensor([[0.11, 3.3, 1.1], [1.1, 0.2, 1.4], [1.1, 2.2, 0.3]], dtype=mstype.float32)


def check_outputs(get_all, get_by_list, original_output, after_hook_output, weight_with_cb_idx,
                  grad_func=double_gradient):
    def check_tuple_output(before, after, tuple_size, grad_idx):
        assert isinstance(before, tuple)
        assert isinstance(after, tuple)
        assert len(before) == len(after) == tuple_size
        for idx in range(tuple_size):
            if idx in grad_idx:
                assert np.allclose(grad_func(before[idx].asnumpy()), after[idx].asnumpy())
            else:
                assert np.allclose(before[idx].asnumpy(), after[idx].asnumpy())

    def check_tensor_output(before, after):
        assert isinstance(before, Tensor)
        assert isinstance(after, Tensor)
        assert np.allclose(before.asnumpy(), after.asnumpy())

    if get_all:
        if get_by_list:
            check_tuple_output(original_output[0], after_hook_output[0], 2, [])
            check_tuple_output(original_output[1], after_hook_output[1], 4, weight_with_cb_idx)
        else:
            check_tuple_output(original_output, after_hook_output, 2, weight_with_cb_idx)
    else:
        if get_by_list:
            check_tuple_output(original_output, after_hook_output, 4, weight_with_cb_idx)
        else:
            check_tensor_output(original_output, after_hook_output)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('get_all', [False, True])
@pytest.mark.parametrize('get_by_list', [False, True])
@pytest.mark.parametrize('x', [input_x])
@pytest.mark.parametrize('y', [input_y])
@pytest.mark.parametrize('forward_hook', [None, param_hook1])
@pytest.mark.parametrize('backward_hook', [None, gradient_hook1])
def test_register_single_layer_hook(get_all, get_by_list, x, y, forward_hook, backward_hook):
    """
    Feature: ALL TO ALL
    Description: Test register hook on one layer only.
    Expectation: The layer of which the hook is registered should apply the hook, while others not.
    Note: WE DID NOT ASSERT WHETHER THE FORWARD hook HAS BEEN RUN HERE.
    """
    context.set_context(mode=context.GRAPH_MODE)
    compile_config.CELL_PARAMETERS_HOOK = '1'

    # get original output
    net1 = Net1()
    net2 = Net2(net1)
    grad_net = GradNet(net2, get_all=get_all, get_by_list=get_by_list)
    original_output = grad_net(x, y)

    # get output after hook registered
    cb_net1 = Net1()
    cb_net2 = Net2(cb_net1)
    cb_grad_net = GradNet(cb_net2, get_all=get_all, get_by_list=get_by_list)
    cb_net1._register_parameters_hook(forward_hook=forward_hook, backward_hook=backward_hook) # pylint:disable=protected-access
    after_hook_output = cb_grad_net(x, y)

    weight_with_cb_idx = [] if backward_hook is None else [2, 3]
    check_outputs(get_all, get_by_list, original_output, after_hook_output, weight_with_cb_idx)
    compile_config.CELL_PARAMETERS_HOOK = ''

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_register_multi_hooks():
    """
    Feature: ALL TO ALL
    Description: Test register more then one hook on a single layer.
    Expectation: Only the last hook should be applied.
    Note: WE DID NOT ASSERT WHETHER THE FORWARD hook HAS BEEN RUN HERE.
    """
    context.set_context(mode=context.GRAPH_MODE)
    compile_config.CELL_PARAMETERS_HOOK = '1'

    net1 = Net1()
    net2 = Net2(net1)
    grad_net = GradNet(net2, get_all=True, get_by_list=True)
    original_output = grad_net(input_x, input_y)

    cb_net1 = Net1()
    cb_net2 = Net2(cb_net1)
    cb_grad_net = GradNet(cb_net2, get_all=True, get_by_list=True)
    cb_net1._register_parameters_hook(forward_hook=param_hook1, backward_hook=gradient_hook1) # pylint:disable=protected-access
    cb_net1._register_parameters_hook(forward_hook=param_hook3, backward_hook=gradient_hook3) # pylint:disable=protected-access
    after_hook_output = cb_grad_net(input_x, input_y)

    check_outputs(True, True, original_output, after_hook_output, [2, 3], quard_gradient)
    compile_config.CELL_PARAMETERS_HOOK = ''

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_register_hook_for_all_cells():
    """
    Feature: ALL TO ALL
    Description: Test register hook for all cells.
    Expectation: All sub cells are registered with hooks/
    Note: WE DID NOT ASSERT WHETHER THE FORWARD hook HAS BEEN RUN HERE.
    """
    context.set_context(mode=context.GRAPH_MODE)
    compile_config.CELL_PARAMETERS_HOOK = '1'

    net1 = Net1()
    net2 = Net2(net1)
    grad_net = GradNet(net2, get_all=True, get_by_list=True)
    original_output = grad_net(input_x, input_y)

    cb_net1 = Net1()
    cb_net2 = Net2(cb_net1)
    cb_grad_net = GradNet(cb_net2, get_all=True, get_by_list=True)
    cb_net2._register_parameters_hook(forward_hook=param_hook1, backward_hook=gradient_hook1, all=True) # pylint:disable=protected-access
    after_hook_output = cb_grad_net(input_x, input_y)

    check_outputs(True, True, original_output, after_hook_output, [0, 1, 2, 3], double_gradient)
    compile_config.CELL_PARAMETERS_HOOK = ''

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('net1_forward, net1_backward, net2_forward, net2_backward',
                         [(param_hook1, gradient_hook1, param_hook1, gradient_hook1),
                          (param_hook1, gradient_hook1, param_hook2, gradient_hook2)])
def test_register_multi_layer_hook(net1_forward, net1_backward, net2_forward, net2_backward):
    """
    Feature: ALL TO ALL
    Description: Test register hook on multi layers.
    Expectation: The layers of which the hook is registered should apply hooks, while others not.
    Note: WE DID NOT ASSERT WHETHER THE FORWARD hook HAS BEEN RUN HERE.
    """
    context.set_context(mode=context.GRAPH_MODE)
    compile_config.CELL_PARAMETERS_HOOK = '1'

    net1 = Net1()
    net2 = Net2(net1)
    grad_net = GradNet(net2, get_all=True, get_by_list=True)
    original_output = grad_net(input_x, input_y)

    cb_net1 = Net1()
    cb_net2 = Net2(cb_net1)
    cb_grad_net = GradNet(cb_net2, get_all=True, get_by_list=True)
    cb_net1._register_parameters_hook(forward_hook=net1_forward, backward_hook=net1_backward) # pylint:disable=protected-access
    cb_net2._register_parameters_hook(forward_hook=net2_forward, backward_hook=net2_backward) # pylint:disable=protected-access
    after_hook_output = cb_grad_net(input_x, input_y)

    check_outputs(True, True, original_output, after_hook_output, [0, 1, 2, 3])
    compile_config.CELL_PARAMETERS_HOOK = ''


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_grad_net_hook():
    """
    Feature: ALL TO ALL
    Description: Test register hook on grad net.
    Expectation: Grad net should run forward hook, but not backward hook.
    Note: Generally, it is not common to register hook on grad net.
    """
    context.set_context(mode=context.GRAPH_MODE)
    compile_config.CELL_PARAMETERS_HOOK = '1'

    net1 = Net1()
    net2 = Net2(net1)
    grad_net = GradNet(net2, get_all=True, get_by_list=True)
    original_output = grad_net(input_x, input_y)

    cb_net1 = Net1()
    cb_net2 = Net2(cb_net1)
    cb_grad_net = GradNet(cb_net2, get_all=True, get_by_list=True)
    cb_grad_net._register_parameters_hook(forward_hook=param_hook1, backward_hook=gradient_hook1) # pylint:disable=protected-access
    after_hook_output = cb_grad_net(input_x, input_y)

    check_outputs(True, True, original_output, after_hook_output, [0, 1, 2, 3])
    compile_config.CELL_PARAMETERS_HOOK = ''

class NetNoParameter(nn.Cell):
    def __init__(self):
        super(NetNoParameter, self).__init__()
        self.matmul = ops.MatMul()

    def construct(self, x, y):
        out = self.matmul(x, y)
        return out


class GradNetNoParameter(nn.Cell):
    def __init__(self, net_no_parameter):
        super(GradNetNoParameter, self).__init__()
        self.net_no_parameter = net_no_parameter
        self.params = ParameterTuple(net_no_parameter.trainable_params())
        self.grad_op = ops.GradOperation(get_all=True, get_by_list=True)

    def construct(self, x, y):
        gradient_function = self.grad_op(self.net_no_parameter, self.params)
        grad = gradient_function(x, y)
        return grad


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_no_parameter_net_hook():
    """
    Feature: ALL TO ALL
    Description: Test register hook on the net without parameters.
    Expectation: Both forward hook and backward hook should not be run.
    Note: WE DID NOT ASSERT WHETHER THE FORWARD hook HAS BEEN RUN HERE.
    """
    context.set_context(mode=context.GRAPH_MODE)
    compile_config.CELL_PARAMETERS_HOOK = '1'

    net_no_param = NetNoParameter()
    grad_net_no_param = GradNetNoParameter(net_no_param)
    original_output = grad_net_no_param(input_x, input_y)

    cb_net_no_param = NetNoParameter()
    cb_grad_net_no_param = GradNetNoParameter(cb_net_no_param)
    net_no_param._register_parameters_hook(forward_hook=param_hook1, backward_hook=gradient_hook1) # pylint:disable=protected-access
    after_hook_output = cb_grad_net_no_param(input_x, input_y)

    assert len(original_output) == len(after_hook_output) == 2
    assert len(original_output[0]) == len(after_hook_output[0]) == 2
    assert np.allclose(original_output[0][0].asnumpy(), after_hook_output[0][0].asnumpy())
    assert np.allclose(original_output[0][1].asnumpy(), after_hook_output[0][1].asnumpy())
    assert not original_output[1] and not after_hook_output[1]
    compile_config.CELL_PARAMETERS_HOOK = ''
