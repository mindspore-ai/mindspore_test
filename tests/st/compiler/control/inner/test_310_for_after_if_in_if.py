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
from tests.mark_utils import arg_mark
from mindspore import context
from mindspore import Tensor, nn
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from mindspore.common.parameter import Parameter

context.set_context(jit_config={"jit_level": "O0"})
grad_all = C.GradOperation(get_all=True)


@arg_mark(plat_marks=['platform_gpu',], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_for_after_if_in_if():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class ForAfterIfInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(4, mstype.int32), name='b')

        def construct(self, x):
            out = self.param_a
            if self.param_a > self.param_b:
                x += 3
                if x > self.param_a:
                    self.param_b += 4
                    x += self.param_a
            self.param_b += 2
            for _ in range(0, 5):
                out += self.param_b
            out *= x
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor(5, mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_after_if_in_if_net = ForAfterIfInIfNet()
    net = GradNet(for_after_if_in_if_net)

    forward_net = ForAfterIfInIfNet()
    graph_forward_res = forward_net(x)
    graph_backward_res = net(x)

    assert graph_forward_res == Tensor(715, mstype.int32)
    assert graph_backward_res == (Tensor(55, mstype.int32),)


def test_for_after_if_in_if_in_vm():
    """
    Feature: Control flow
    Description: Test control flow in graph mode.
    Expectation: No exception.
    """
    class ForAfterIfInIfNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_a = Parameter(Tensor(5, mstype.int32), name='a')
            self.param_b = Parameter(Tensor(4, mstype.int32), name='b')

        def construct(self, x):
            out = self.param_a
            while x < 0:
                x += 1
            if self.param_a > self.param_b:
                x += 3
                if x > self.param_a:
                    self.param_b += 4
                    x += self.param_a
            self.param_b += 2
            for _ in range(0, 5):
                out += self.param_b
            out *= x
            return out

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, *inputs):
            return grad_all(self.net)(*inputs)

    x = Tensor(5, mstype.int32)

    # graph mode
    context.set_context(mode=context.GRAPH_MODE)
    for_after_if_in_if_net = ForAfterIfInIfNet()
    net = GradNet(for_after_if_in_if_net)

    forward_net = ForAfterIfInIfNet()
    graph_forward_res = forward_net(x)
    graph_backward_res = net(x)

    assert graph_forward_res == Tensor(715, mstype.int32)
    assert graph_backward_res == (Tensor(55, mstype.int32),)
