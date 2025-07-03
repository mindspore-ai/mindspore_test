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

import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Symbol
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.parallel.shard import Layout
from parallel.utils.utils import ParallelValidator, compile_net

class Net(Cell):
    def __init__(self, diagonal=0, strategy=None):
        super().__init__()
        if strategy:
            self.triu = P.Triu(diagonal).shard(strategy)
        else:
            self.triu = P.Triu(diagonal)

    def construct(self, x):
        out = self.triu(x)
        return out

def test_semi_auto_parallel_triu_0():
    """
    Feature: test triu semi auto parallel
    Description: semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=1)
    x = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
    diagonal = 2
    net = Net(diagonal)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('Triu-0', ['TupleGetItem-0', 2])

def test_semi_auto_parallel_triu_1():
    """
    Feature: test triu semi auto parallel
    Description: semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=4)
    x = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
    diagonal = 0
    strategy = ((1, 2, 4),)
    net = Net(diagonal, strategy)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('Triu-0', ['Reshape-1', 8])

def test_semi_auto_parallel_triu_2():
    """
    Feature: test triu semi auto parallel
    Description: semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=7)
    x = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
    diagonal = -2
    strategy = ((2, 1, 4),)
    net = Net(diagonal, strategy)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('Triu-0', ['Reshape-1', -26])

def test_semi_auto_parallel_triu_3():
    """
    Feature: test triu semi auto parallel
    Description: semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=4)
    x = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
    diagonal = 0
    layout = Layout((1, 2, 4), ("ap", "bp", "cp"))
    strategy = (layout("ap", "bp", "cp"),)
    net = Net(diagonal, strategy)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('Triu-0', ['Reshape-1', 8])

def test_semi_auto_parallel_triu_4():
    """
    Feature: test triu semi auto parallel
    Description: semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=7)
    x = Tensor(np.ones([16, 64, 32]), dtype=ms.float32)
    diagonal = -2
    layout = Layout((2, 1, 4), ("ap", "bp", "cp"))
    strategy = (layout("ap", "bp", "cp"),)
    net = Net(diagonal, strategy)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('Triu-0', ['Reshape-1', -26])

def test_semi_auto_parallel_triu_5():
    """
    Feature: test triu semi auto parallel with dynamic shape
    Description: semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=4)
    diagonal = 0
    strategy = ((1, 2, 4),)
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=4)
    x = Tensor(shape=[16, s1, s2], dtype=ms.float32)
    net = Net(diagonal, strategy)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('Triu-0', ['Reshape-1', 'ScalarMax-0'])

def test_semi_auto_parallel_triu_6():
    """
    Feature: test triu semi auto parallel with dynamic shape
    Description: semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=4)
    diagonal = 0
    layout = Layout((1, 2, 4), ("ap", "bp", "cp"))
    strategy = (layout("ap", "bp", "cp"),)
    s1 = Symbol(divisor=2)
    x = Tensor(shape=[16, s1, 32], dtype=ms.float32)
    net = Net(diagonal, strategy)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('Triu-0', ['Reshape-1', 'ScalarMax-0'])

def test_semi_auto_parallel_triu_7():
    """
    Feature: test triu semi auto parallel with dynamic shape
    Description: semi auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=4)
    diagonal = 0
    strategy = ((2, 1, 4),)
    s1 = Symbol(divisor=4)
    x = Tensor(shape=[16, 64, s1], dtype=ms.float32)
    net = Net(diagonal, strategy)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('Triu-0', ['Reshape-1', 'ScalarMax-0'])
