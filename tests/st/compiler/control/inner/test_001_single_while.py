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
import pytest
from tests.mark_utils import arg_mark
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import Tensor, jit
from mindspore.ops import composite as C
from mindspore import context

context.set_context(mode=context.GRAPH_MODE)
context.set_context(jit_config={"jit_level": "O0"})


class ForwardNet(nn.Cell):
    def construct(self, x, y):
        y = y + 10
        while x < y:
            x = (x + 2) * (y - 9)
            y = y + 2
        x = x + 5
        return x


class BackwardNet(nn.Cell):
    def __init__(self, forward_net):
        super(BackwardNet, self).__init__()
        self.forward_net = forward_net
        self.grad = C.GradOperation()

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
    c1 = Tensor([0], mstype.int32)
    c2 = Tensor([0], mstype.int32)
    expect = Tensor([75], mstype.int32)
    forward_net = ForwardNet()
    output = forward_net(c1, c2)
    assert expect == output


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_backward():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    c1 = Tensor([0], mstype.int32)
    c2 = Tensor([0], mstype.int32)
    expect = Tensor([15], mstype.int32)
    forward_net = ForwardNet()
    backward_net = BackwardNet(forward_net)
    output = backward_net(c1, c2)
    assert expect == output


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_single_while():
    """
    Feature: The else branches of while loops aren't supported.
    Description: The else branches of while loops aren't supported.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_while(x, y):
        while x > y:
            y += x
            break
        else:
            y = x + 6
        return y

    with pytest.raises(RuntimeError, match="The 'while...else...' statement is not supported now."):
        input_x = Tensor([0], mstype.int32)
        input_y = Tensor([2], mstype.int32)
        res = control_flow_while(input_x, input_y)
        print("res:", res)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_single_while_tensor():
    """
    Feature: The else branches of while loops aren't supported.
    Description: The else branches of while loops aren't supported.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_while_tensor(x):
        while x:
            x -= 1
        return x

    x = Tensor(1)
    out = control_flow_while_tensor(x)
    assert out == 0
