# Copyright 2020 Huawei Technologies Co., Ltd
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
import numpy as np
from tests.mark_utils import arg_mark
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import composite as C
from mindspore import context
from mindspore.common.parameter import Parameter

context.set_context(mode=context.GRAPH_MODE)
context.set_context(jit_config={"jit_level": "O0"})


class ForwardNet(nn.Cell):
    def __init__(self, max_cycles=10):
        super(ForwardNet, self).__init__()
        self.max_cycles = max_cycles
        self.i = Tensor(np.array(0), mstype.int32)
        self.zero = Tensor(np.array(0), mstype.int32)
        self.weight = Parameter(Tensor(np.array(0), mstype.int32))

    def construct(self, x, y):
        i = self.i
        out = self.zero
        while i < self.max_cycles:
            self.weight = i
            if out <= 20:
                out = x * y + out
                self.weight = out
            i = i + 1
        i = self.i
        while i < self.max_cycles:
            self.weight = i
            out = out + 10
            i = i + 1
        out1 = self.weight + 1
        return out, out1


class BackwardNet(nn.Cell):
    def __init__(self, net):
        super(BackwardNet, self).__init__(auto_prefix=False)
        self.forward_net = net
        self.grad = C.GradOperation(get_all=True)

    def construct(self, *inputs):
        grads = self.grad(self.forward_net)(*inputs)
        return grads


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_forward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    graph_forward_net = ForwardNet(max_cycles=10)
    graph_mode_out = graph_forward_net(x, y)

    assert graph_mode_out == (Tensor(np.array(121), mstype.int32), Tensor(np.array(10), mstype.int32))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_backward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    x = Tensor(np.array(1), mstype.int32)
    y = Tensor(np.array(3), mstype.int32)
    # Graph Mode
    context.set_context(mode=context.GRAPH_MODE)
    graph_forward_net = ForwardNet(max_cycles=10)
    graph_backward_net = BackwardNet(graph_forward_net)
    graph_mode_grads = graph_backward_net(x, y)

    assert graph_mode_grads == (Tensor(np.array(21), mstype.int32), Tensor(np.array(7), mstype.int32))
