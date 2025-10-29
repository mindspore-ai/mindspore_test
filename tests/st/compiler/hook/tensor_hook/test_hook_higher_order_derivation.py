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
"""Test hook with higher order derivation"""

import pytest
import numpy as np
import torch
import mindspore as ms
from mindspore import nn, Tensor, ops
from tests.mark_utils import arg_mark


class MulNetTorch(torch.nn.Module):
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

    def double_fn(self, grad):
        return 2 * grad


def double_fn(grad):
    return grad * 2


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mul = MulNet()
        self.param = ms.Parameter(ms.Tensor([10.0, 10.0, 10.0], ms.float32), name="w1")

    @ms.jit
    def construct(self, x):
        xx = self.mul(x, x)
        xxp = self.mul(xx, self.param)
        xxpp = self.mul(xxp, self.param)
        return xxpp


class Mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mul = MulNetTorch()
        self.param = torch.nn.Parameter(torch.tensor([10.0, 10.0, 10.0], dtype=torch.float32))

    def forward(self, x):
        xx = self.mul(x, x)
        xxp = self.mul(xx, self.param)
        xxpp = self.mul(xxp, self.param)
        return xxpp


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_hook_higher_order_derivation_001(mode):
    """
    Feature: Hook
    Description: Test hook with higher order derivation
    Expectation: No exception.
    """
    ms.set_context(mode=mode)
    net = Net()
    mod = Mod()
    input_np = np.array([2.0, 3.0, 4.0]).astype(np.float32)
    input_ms = Tensor(input_np)

    input_ms.register_hook(double_fn)
    out_ms = net(input_ms)
    first_grad_net = ops.grad(net, grad_position=None, weights=net.param)
    second_grad_net = ops.grad(first_grad_net, grad_position=None, weights=net.param)
    grad_ms = second_grad_net(input_ms)
    grad_ms1 = first_grad_net(input_ms)

    x_torch = torch.tensor(input_np, dtype=torch.float32, requires_grad=True)
    x_torch.register_hook(double_fn)
    y = mod(x_torch)
    sens = torch.ones_like(y)
    first_grad = torch.autograd.grad(y, mod.param, grad_outputs=sens, create_graph=True)[0]
    second_grad = torch.autograd.grad(first_grad, mod.param, grad_outputs=torch.ones_like(first_grad))[0]

    assert np.allclose(y.detach().numpy(), out_ms.asnumpy(), 0.001, 0.001)
    assert np.allclose(second_grad.detach().numpy(), grad_ms.asnumpy(), 0.001, 0.001)
    assert np.allclose(first_grad.detach().numpy(), grad_ms1.asnumpy(), 0.001, 0.001)
