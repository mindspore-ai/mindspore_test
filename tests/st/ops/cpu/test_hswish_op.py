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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True)
        self.network = network

    @jit
    def construct(self, input_):
        return self.grad(self.network)(input_)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.HSwish = P.HSwish()

    def construct(self, x):
        return self.HSwish(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_net():
    x = np.array([-1, -2, 0, 2, 1]).astype(np.float32)
    hswish = Net()
    y = hswish(Tensor(x))
    expect = np.array([-0.33333334, -0.33333334, 0., 1.6666666, 0.6666667]).astype(np.float32)
    assert np.all(y.asnumpy() == expect)
    backword_net = Grad(Net())
    output = backword_net(Tensor(x))
    print(len(output))
    print(output[0].asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_hswish_cpu_dynamic_shape():
    """
    Feature: test HSwish op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    net = Net()
    x_dyn = Tensor(shape=[None, 32], dtype=mindspore.float32)
    net.set_inputs(x_dyn)
    x = np.random.randn(3, 32)
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (3, 32)
    assert output.asnumpy().shape == expect_shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_hswish_grad_cpu_dynamic_shape():
    """
    Feature: test HSwishGrad op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    net = Net()
    grad = Grad(net)
    dy_dyn = Tensor(shape=[None, 32], dtype=mindspore.float32)
    grad.set_inputs(dy_dyn)
    dy = np.random.randn(3, 32)
    output = grad(Tensor(dy, mindspore.float32))
    expect_shape = (3, 32)
    assert output[0].asnumpy().shape == expect_shape
