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
"""Test cell hook"""

import pytest
import numpy as np
import torch
from torch.nn import Module
import mindspore as ms
from mindspore import nn, Tensor, ops
from tests.mark_utils import arg_mark


class MulNetTorch(Module):
    def __init__(self):
        super().__init__()
        self.mul = torch.mul
        self.relu = torch.nn.ReLU()

    def forward(self, x, y):
        x = self.mul(x, y)
        x = self.relu(x)
        return x


class MulNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = ops.Mul()
        self.relu = nn.ReLU()

    def construct(self, x, y):
        x = self.mul(x, y)
        x = self.relu(x)
        return x


class GradOfAllInputs(nn.Cell):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.grad_op = ops.GradOperation(get_all=True, sens_param=True)

    def construct(self, *inputs):
        grad_net = self.grad_op(self.net)
        return grad_net(*inputs)


def double_fn(cell_id, inputs):
    modified_inputs = (inputs[0] * 2, inputs[1] * 2)
    return modified_inputs


def double_back(cell, grad_input, grad_output):
    return grad_input[0] + grad_output[0], grad_input[1] + grad_output[0]


def double_pback(cell, grad_output):
    return tuple(g * 2 for g in grad_output)


def forward_hook_fn(cell, inputs, output):
    return output + inputs[0] * inputs[1]


def compare_with_torch(ms_net, torch_net):
    input1_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input2_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)

    input1_ms = Tensor(input1_np)
    input2_ms = Tensor(input2_np)

    ms_net.set_grad()
    out_ms = ms_net(input1_ms, input2_ms)
    grad_net = GradOfAllInputs(ms_net)
    grad_net.set_train()
    input_ms_grad = grad_net(input1_ms, input2_ms, out_ms)

    input1_torch = torch.from_numpy(input1_np)
    input2_torch = torch.from_numpy(input2_np)
    input1_torch.requires_grad = True
    input2_torch.requires_grad = True

    out_torch = torch_net(input1_torch, input2_torch)
    out_torch.backward(out_torch)

    assert np.allclose(out_torch.detach().numpy(), out_ms.asnumpy(), 0.00001, 0.00001)
    assert np.allclose(input1_torch.grad.numpy(), input_ms_grad[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(input2_torch.grad.numpy(), input_ms_grad[1].asnumpy(), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_jit_cell_hook(mode):
    """
    Feature: Hook
    Description: Test cell hook with jit
    Expectation: No exception.
    """
    class Mod(Module):
        def __init__(self):
            super().__init__()
            self.mul = MulNetTorch()
            self.handle71 = self.mul.register_forward_pre_hook(double_fn)
            self.handle72 = self.mul.register_forward_hook(forward_hook_fn)
            self.handle73 = self.mul.register_full_backward_pre_hook(double_pback)
            self.handle74 = self.mul.register_full_backward_hook(double_back)

        def forward(self, x, y):
            x = x + x
            x = self.mul(x, y)
            return x

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = MulNet()
            self.handle75 = self.mul.register_forward_pre_hook(double_fn)
            self.handle76 = self.mul.register_forward_hook(forward_hook_fn)
            self.handle77 = self.mul.register_backward_pre_hook(double_pback)
            self.handle78 = self.mul.register_backward_hook(double_back)

        def construct(self, x, y):
            x = x + x
            x = self.mul(x, y)
            return x

    ms.set_context(mode=mode)
    ms_net = Net()
    torch_net = Mod()
    compare_with_torch(ms_net, torch_net)
