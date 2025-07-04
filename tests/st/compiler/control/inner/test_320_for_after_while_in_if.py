# Copyright 2021-2024 Huawei Technologies Co., Ltd
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
from mindspore import context
from mindspore import Tensor, nn
from mindspore.common.parameter import Parameter
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

context.set_context(jit_config={"jit_level": "O0"})
grad_all = C.GradOperation(get_all=True)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_for_after_while_in_if_01():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class ForAfterWhileInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.relu = nn.ReLU()
            self.softmax = nn.Softmax()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.div = P.Div()
            self.assign = P.Assign()
            param_a = np.full((1,), 5, dtype=np.float32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 2, dtype=np.float32)
            self.param_b = Parameter(Tensor(param_b), name='b')
            param_c = np.full((1,), 16, dtype=np.float32)
            self.param_c = Parameter(Tensor(param_c), name='c')

        def construct(self, x, y):
            self.assign(self.param_a, x + self.param_a)
            y = self.add(y, self.param_b)

            if self.param_b != y - self.param_a:
                self.param_c = self.div(self.param_c, self.param_b)
                while self.param_a > x:
                    self.param_c = self.param_a + 2
                    x = x + 1
                y = self.softmax(self.param_c)
                self.param_b = self.sub(y, self.param_b)

            x = self.mul(self.param_b, self.param_c)

            for _ in range(0, 4):
                x = self.sub(x, 3)
                y = y + self.param_b

            self.param_a = x + y
            z = self.relu(y + self.param_a)
            return z

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([11], mstype.int32)
    y = Tensor([7], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_after_while_in_if_net = ForAfterWhileInIfNet()
    net = GradNet(for_after_while_in_if_net)

    forward_net = ForAfterWhileInIfNet()
    graph_forward_res = forward_net(x, y)
    graph_backward_res = net(x, y)

    assert graph_forward_res == Tensor([0], mstype.float32)
    assert graph_backward_res == (Tensor([0], mstype.int32), Tensor([0], mstype.int32))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_for_after_while_in_if_02():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class ForAfterWhileInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.mul = P.Mul()
            self.add = P.Add()
            self.sub = P.Sub()
            self.assign = P.Assign()
            param_a = np.full((1,), 5, dtype=np.int32)
            self.param_a = Parameter(Tensor(param_a), name='a')
            param_b = np.full((1,), 2, dtype=np.int32)
            self.param_b = Parameter(Tensor(param_b), name='b')
            param_c = np.full((1,), 11, dtype=np.int32)
            self.param_c = Parameter(Tensor(param_c), name='c')

        def construct(self, x, y):
            self.assign(self.param_a, x + self.param_a)
            y = self.add(y, self.param_b)
            if (self.param_b > (y - self.param_a)) and (self.param_b != self.param_a):
                x = y - self.param_a - self.param_b
                while self.param_a >= x:
                    self.assign(self.param_c, self.param_a + 2)
                    x = x + 2
                self.param_b = self.sub(y, self.param_b)
            x = self.mul(self.param_b, self.param_c)
            for _ in range(0, 4):
                self.assign(self.param_b, y + self.param_b - x)
                y = x + self.param_a - self.param_b
            return y

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor([11], mstype.int32)
    y = Tensor([7], mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_after_while_in_if_net = ForAfterWhileInIfNet()
    net = GradNet(for_after_while_in_if_net)

    forward_net = ForAfterWhileInIfNet()
    graph_forward_res = forward_net(x, y)
    graph_backward_res = net(x, y)

    assert graph_forward_res == Tensor([126], mstype.int32)
    assert graph_backward_res == (Tensor([0], mstype.int32), Tensor([0], mstype.int32))
