# Copyright 2021 Huawei Technologies Co., Ltd
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
"""test jvp in graph mode"""
import numpy as np
import pytest
import torch
from torch.autograd.functional import vjp
from mindspore import nn, context, ops, Tensor
from mindspore.nn.grad import Vjp
from tests.mark_utils import arg_mark


class SingleInputNet(nn.Cell):
    def construct(self, x):
        return x**3


class MultipleInputsOutputNet(nn.Cell):
    def construct(self, x, y):
        return 2*x, y**3


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_single_input_graph(mode):
    """
    Features: Class Vjp.
    Description: Test whenther Vjp can calculate backward-mode diff correctly.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = SingleInputNet()
    expect_primal = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    primal, grad = Vjp(net)(x, v)
    assert np.allclose(primal.asnumpy(), expect_primal.asnumpy())
    assert np.allclose(grad.asnumpy(), expect_grad.asnumpy())



@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_vjp_multiple_inputs_default_v_graph(mode):
    """
    Features: Class Vjp.
    Description: Test whenther Vjp can calculate backward-mode diff correctly.
    Expectation: No exception.
    """
    context.set_context(mode=mode)
    x = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    y = Tensor(np.array([[1, 2], [3, 4]]).astype(np.float32))
    v = Tensor(np.array([[1, 1], [1, 1]]).astype(np.float32))
    net = MultipleInputsOutputNet()
    expect_primal_0 = Tensor(np.array([[2, 4], [6, 8]]).astype(np.float32))
    expect_primal_1 = Tensor(np.array([[1, 8], [27, 64]]).astype(np.float32))
    expect_grad_0 = Tensor(np.array([[2, 2], [2, 2]]).astype(np.float32))
    expect_grad_1 = Tensor(np.array([[3, 12], [27, 48]]).astype(np.float32))
    primal, grad = Vjp(net)(x, y, (v, v))
    assert isinstance(primal, tuple)
    assert len(primal) == 2
    assert np.allclose(primal[0].asnumpy(), expect_primal_0.asnumpy())
    assert np.allclose(primal[1].asnumpy(), expect_primal_1.asnumpy())
    assert isinstance(grad, tuple)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), expect_grad_0.asnumpy())
    assert np.allclose(grad[1].asnumpy(), expect_grad_1.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_vjp_in2_out2_param():
    """
    Feature: vjp
    Description: Test vjp and compare with torch
    Expectation: No exception.
    """
    class MsConvRelu(nn.Cell):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 8, 3, 3, "valid")
            self.relu = nn.ReLU()

        def construct(self, x, y):
            a = self.conv(x)
            b = self.conv(y)
            c = a + b
            return self.relu(c), c

    class TcConvRelu:
        def __init__(self):
            self.conv = torch.nn.Conv2d(4, 8, 3, 3, bias=False)
            self.relu = torch.nn.ReLU()

        def construct(self, x, y):
            a = self.conv(x)
            b = self.conv(y)
            c = a + b
            return self.relu(c), c

    class MSVjp(nn.Cell):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def construct(self, *args):
            out, grad_net = ops.vjp(self.net, *args[:-1])
            grad = grad_net(args[-1])
            return out, grad

    context.set_context(mode=context.GRAPH_MODE)
    ms_net = MsConvRelu()
    weight = ms_net.conv.weight.asnumpy()
    tc_net = TcConvRelu()
    tc_net.conv.register_parameter('weight', torch.nn.Parameter(torch.from_numpy(weight)))
    x = np.random.rand(4, 4, 4, 4).astype(np.float32)
    y = np.random.rand(4, 4, 4, 4).astype(np.float32)
    sense_shape = ((4, 8, 1, 1), (4, 8, 1, 1))

    ms_inputs = (Tensor(x), Tensor(y))
    tc_inputs = (torch.tensor(x, requires_grad=True), torch.tensor(y, requires_grad=True))
    usenses = [np.random.rand(*shape).astype(np.float32) for shape in sense_shape]
    # pylint: disable=consider-using-generator
    ms_sense = tuple([Tensor(v) for v in usenses])
    tc_sense = tuple([torch.tensor(v) for v in usenses])
    ms_out, ms_grad = MSVjp(ms_net)(*ms_inputs, ms_sense)
    tc_out, tc_grad = vjp(tc_net.construct, tc_inputs, tc_sense)
    if isinstance(ms_out, tuple):
        for m, t in zip(ms_out, tc_out):
            assert np.allclose(m.asnumpy(), t.detach().numpy(), 0.001, 0.001)
    else:
        assert np.allclose(ms_out.asnumpy(), tc_out.detach().numpy(), 0.001, 0.001)
    if isinstance(ms_grad, tuple):
        for m, t in zip(ms_grad, tc_grad):
            assert np.allclose(m.asnumpy(), t.detach().numpy(), 0.001, 0.001)
    else:
        assert np.allclose(ms_grad.asnumpy(), tc_grad[0].detach().numpy(), 0.001, 0.001)
