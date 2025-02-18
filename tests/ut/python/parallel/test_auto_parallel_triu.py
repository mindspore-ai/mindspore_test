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
from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P
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
