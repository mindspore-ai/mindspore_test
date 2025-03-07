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
import subprocess
import shutil
import pytest
import numpy as np
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, Parameter
from tests.mark_utils import arg_mark

def hook_double(grad):
    return grad * 2

def hook_triple(grad):
    return grad * 3

def hook_mul_5(grad):
    return grad * 5

def hook_print(grad):
    print("grad:", grad)

np_weight0 = np.array([1.0, 2.0, 3.0])
np_weight1 = np.array([4.0, 5.0, 6.0])
np_input_x = np.array([7.0, 8.0, 9.0])

class GroundNet(nn.Cell):
    def __init__(self):
        super(GroundNet, self).__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x = x * self.weight0
        out = x * self.weight1
        return out

class OneTensorOneHookNet(nn.Cell):
    def __init__(self):
        super(OneTensorOneHookNet, self).__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x = x * self.weight0
        x.register_hook(hook_double)
        out = x * self.weight1
        return out

class OneTensorMultiHookNet(nn.Cell):
    def __init__(self):
        super(OneTensorMultiHookNet, self).__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x = x * self.weight0
        x.register_hook(hook_double)
        x.register_hook(hook_triple)
        out = x * self.weight1
        return out

class MultiTensorMultiHookNet(nn.Cell):
    def __init__(self):
        super(MultiTensorMultiHookNet, self).__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x = x * self.weight0
        x.register_hook(hook_double)
        x.register_hook(hook_triple)
        y = x * self.weight1
        y.register_hook(hook_double)
        y.register_hook(hook_triple)
        out = y
        return out

class HookPrintNet(nn.Cell):
    def __init__(self):
        super(HookPrintNet, self).__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x = x * self.weight0
        x.register_hook(hook_print)
        out = x * self.weight1
        return out

class HookInJITNet(nn.Cell):
    def __init__(self):
        super(HookInJITNet, self).__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")

    @ms.jit
    def hook(self, x):
        x.register_hook(hook_double)
        return x

    def construct(self, x):
        x = x * self.weight0
        x = self.hook(x)
        out = x * self.weight1
        return out

class NeedReorderHookStmtNet(nn.Cell):
    def __init__(self):
        super(NeedReorderHookStmtNet, self).__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")

    def construct(self, x):
        x.register_hook(hook_double)
        x = x * self.weight0 + x - x
        x.register_hook(hook_triple)
        x = x * self.weight1 - x + x
        x.register_hook(hook_mul_5)
        return x

ground_input_x = Tensor(np_input_x, ms.float32)
ground_net = GroundNet()
ground_grad_op = ops.GradOperation(get_all=True, get_by_list=True)
ground_grad_net = ground_grad_op(ground_net, ground_net.trainable_params())
ground_output = ground_grad_net(ground_input_x)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_one_hook():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register one hook on one tensor.
    Expectation: The grad of tensor is changed by hook.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_x = Tensor(np_input_x, ms.float32)
    net = OneTensorOneHookNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    output = grad_net(input_x)
    output_x_grad = output[0][0].asnumpy()
    output_weight0_grad = output[1][0].asnumpy()
    output_weight1_grad = output[1][1].asnumpy()
    expected_x_grad = hook_double(ground_output[0][0]).asnumpy()
    expected_weight0_grad = hook_double(ground_output[1][0]).asnumpy()
    expected_weight1_grad = ground_output[1][1].asnumpy()

    assert np.allclose(output_x_grad, expected_x_grad)
    assert np.allclose(output_weight0_grad, expected_weight0_grad)
    assert np.allclose(output_weight1_grad, expected_weight1_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_one_tensor_multi_hook():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register multi hooks on one tensor.
    Expectation: The grad of the tensor is changed by hooks sequentially.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_x = Tensor(np_input_x, ms.float32)
    net = OneTensorMultiHookNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    output = grad_net(input_x)
    output_x_grad = output[0][0].asnumpy()
    output_weight0_grad = output[1][0].asnumpy()
    output_weight1_grad = output[1][1].asnumpy()
    expected_x_grad = hook_double(hook_triple(ground_output[0][0])).asnumpy()
    expected_weight0_grad = hook_double(hook_triple(ground_output[1][0])).asnumpy()
    expected_weight1_grad = ground_output[1][1].asnumpy()

    assert np.allclose(output_x_grad, expected_x_grad)
    assert np.allclose(output_weight0_grad, expected_weight0_grad)
    assert np.allclose(output_weight1_grad, expected_weight1_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_multi_tensor_multi_hook():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register multi hooks on multi tensors.
    Expectation: The grad of each tensor is changed by its registered hooks sequentially.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    input_x = Tensor(np_input_x, ms.float32)
    net = MultiTensorMultiHookNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    output = grad_net(input_x)
    output_x_grad = output[0][0].asnumpy()
    output_weight0_grad = output[1][0].asnumpy()
    output_weight1_grad = output[1][1].asnumpy()
    expected_x_grad = hook_double(hook_triple(hook_double(hook_triple(ground_output[0][0])))).asnumpy()
    expected_weight0_grad = hook_double(hook_triple(hook_double(hook_triple(ground_output[1][0])))).asnumpy()
    expected_weight1_grad = hook_double(hook_triple(ground_output[1][1])).asnumpy()

    assert np.allclose(output_x_grad, expected_x_grad)
    assert np.allclose(output_weight0_grad, expected_weight0_grad)
    assert np.allclose(output_weight1_grad, expected_weight1_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hook_no_return():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register no return hook on one tensor.
    Expectation: The grad of the tensor is not changed and the hook is applied(by check ir file).
    """
    save_graphs_path = "./test_tensor_hook_inside_graph_no_return"
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path=save_graphs_path)

    input_x = Tensor(np_input_x, ms.float32)
    net = HookPrintNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    output = grad_net(input_x)
    output_x_grad = output[0][0].asnumpy()
    output_weight0_grad = output[1][0].asnumpy()
    output_weight1_grad = output[1][1].asnumpy()
    expected_x_grad = ground_output[0][0].asnumpy()
    expected_weight0_grad = ground_output[1][0].asnumpy()
    expected_weight1_grad = ground_output[1][1].asnumpy()

    assert np.allclose(output_x_grad, expected_x_grad)
    assert np.allclose(output_weight0_grad, expected_weight0_grad)
    assert np.allclose(output_weight1_grad, expected_weight1_grad)

    para = 'Print("grad:'
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, os.path.join(save_graphs_path, "*validate*.ir"))],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "1"

    if os.path.exists(save_graphs_path):
        shutil.rmtree(save_graphs_path)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_need_reorder_hook_stmt_net():
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register hook after tensor is used.
    Expectation: The grad of the tensor is changed (equal to reorder hook stmt before tensor is used).
    """
    context.set_context(mode=context.GRAPH_MODE, save_graphs=False)

    input_x = Tensor(np_input_x, ms.float32)
    net = NeedReorderHookStmtNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    output = grad_net(input_x)
    output_x_grad = output[0][0].asnumpy()
    output_weight0_grad = output[1][0].asnumpy()
    output_weight1_grad = output[1][1].asnumpy()
    expected_x_grad = hook_mul_5(hook_triple(hook_double(ground_output[0][0]))).asnumpy()
    expected_weight0_grad = hook_mul_5(hook_triple(ground_output[1][0])).asnumpy()
    expected_weight1_grad = hook_mul_5(ground_output[1][1]).asnumpy()

    assert np.allclose(output_x_grad, expected_x_grad)
    assert np.allclose(output_weight0_grad, expected_weight0_grad)
    assert np.allclose(output_weight1_grad, expected_weight1_grad)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_hook_in_jit(mode):
    """
    Feature: Tensor.register_hook(hook_fn) inside graph.
    Description: Test register hook inside jit wrapper
    Expectation: The grad of tensor is changed by hook.
    """
    context.set_context(mode=mode, device_target="CPU")
    input_x = Tensor(np_input_x, ms.float32)
    net = HookInJITNet()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    output = grad_net(input_x)
    output_x_grad = output[0][0].asnumpy()
    output_weight0_grad = output[1][0].asnumpy()
    output_weight1_grad = output[1][1].asnumpy()
    expected_x_grad = hook_double(ground_output[0][0]).asnumpy()
    expected_weight0_grad = hook_double(ground_output[1][0]).asnumpy()
    expected_weight1_grad = ground_output[1][1].asnumpy()

    assert np.allclose(output_x_grad, expected_x_grad)
    assert np.allclose(output_weight0_grad, expected_weight0_grad)
    assert np.allclose(output_weight1_grad, expected_weight1_grad)
