# Copyright 2023 Huawei Technologies Co., Ltd
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
import random

import mindspore.nn as nn
from mindspore import context, jit
from mindspore.common import mutable
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations import _sequence_ops as seq
from sequence_help import TupleFactory, context_prepare

context.set_context(mode=context.GRAPH_MODE)
context_prepare()


class MaxNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.func = seq.SequenceMax()

    def construct(self, x):
        return self.func(x)


class MinNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.func = seq.SequenceMin()

    def construct(self, x):
        return self.func(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_seq_max():
    """
    Feature: test sequence max op
    Description: first input is dynamic sequence
    Expectation: the result match with tuple result
    """

    def func(x):
        return max(x)

    net_ms = MaxNet()
    input_x = tuple([random.randint(-1000, 1000) for i in range(100)])
    fact = TupleFactory(net_ms, func, (input_x,))
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_seq_min():
    """
    Feature: test sequence min op
    Description: first input is dynamic sequence
    Expectation: the result match with tuple result
    """

    def func(x):
        return min(x)

    net_ms = MinNet()
    input_x = tuple([random.randint(-1000, 1000) for i in range(100)])
    fact = TupleFactory(net_ms, func, (input_x,))
    fact.forward_cmp()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_seq_max_grad():
    """
    Feature: test sequence max grad op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = mutable((1, 2, 3), True)
    dout = mutable(2)
    net = MaxNet()
    grad_func = jit(GradOperation(get_all=True, sens_param=True)(net), backend="ms_backend")
    print("grad=:", grad_func(x, dout))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_seq_min_grad():
    """
    Feature: test sequence min grad op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = mutable((1, 2, 3), True)
    dout = mutable(2)
    net = MinNet()
    grad_func = jit(GradOperation(get_all=True, sens_param=True)(net), backend="ms_backend")
    print("grad=:", grad_func(x, dout))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_seq_min_grad_mutable():
    """
    Feature: test sequence min grad op
    Description: two inputs are dynamic sequence
    Expectation: the result match with tuple result
    """
    x = (mutable(1), 2, 3)
    dout = mutable(2)
    net = MinNet()
    grad_func = jit(GradOperation(get_all=True, sens_param=True)(net), backend="ms_backend")
    print("grad=:", grad_func(x, dout))
