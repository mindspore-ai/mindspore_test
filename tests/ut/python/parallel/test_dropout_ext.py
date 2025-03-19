# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore.ops.auto_generate import dropout_ext_op
from parallel.utils.utils import compile_net, ParallelValidator


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        dropout_ext_strategy = strategy2 + ((), ()) if strategy2 is not None else strategy2
        self.mul = P.Mul().shard(strategy1)
        self.dropout1 = dropout_ext_op.shard(dropout_ext_strategy)
        self.relu1 = P.ReLU().shard(strategy2)
        self.dropout2 = dropout_ext_op.shard(dropout_ext_strategy)
        self.relu2 = P.ReLU().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")
        self.seed = Parameter(Tensor(42))
        self.offset = Parameter(Tensor(2))

    def construct(self, x):
        out = self.mul(x, self.mul_weight)
        out, _ = self.dropout1(out, 0.5, ms.Tensor(1), ms.Tensor(1))
        out = self.relu1(out)
        out, _ = self.dropout2(out, 0.6, self.seed, self.offset)
        out = self.relu2(out)
        return out

_x = Tensor(np.ones([128, 64]), dtype=ms.float32)
_w1 = Tensor(np.ones([128, 64]), dtype=ms.float32)


def test_dropout_ext_auto_parallel():
    """
    Feature: test dropout_ext when auto parallel
    Description: auto_parallel
    Expectation: success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation", device_num=16,
                                      global_rank=0)
    net = Net(_w1)
    compile_net(net, _x)


def test_dropout_ext_data_parallel():
    """
    Feature: test dropout_ext when data parallel
    Description: semi_auto_parallel, dp=16, mp=1
    Expectation: success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1), (16, 1))
    strategy2 = ((16, 1),)
    net = Net(_w1, strategy1, strategy2)
    phase = compile_net(net, _x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_strategy("DropoutExt-op0", [[16, 1], [], []])
    assert validator.check_node_strategy("DropoutExt-op1", [[16, 1], [], []])


def test_dropout_ext_model_parallel():
    """
    Feature: test dropout_ext when model parallel
    Description: semi_auto_parallel, dp=1, mp=16
    Expectation: success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 16), (1, 16))
    strategy2 = ((1, 16),)
    net = Net(_w1, strategy1, strategy2)
    phase = compile_net(net, _x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_strategy("DropoutExt-op0", [[1, 16], [], []])
    assert validator.check_node_strategy("DropoutExt-op1", [[1, 16], [], []])


def test_dropout_ext_mixed_parallel():
    """
    Feature: test dropout_ext when mixed parallel
    Description: semi_auto_parallel, dp=4, mp=4
    Expectation: success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 4), (4, 4))
    strategy2 = ((4, 4),)
    net = Net(_w1, strategy1, strategy2)
    phase = compile_net(net, _x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_strategy("DropoutExt-op0", [[4, 4], [], []])
    assert validator.check_node_strategy("DropoutExt-op1", [[4, 4], [], []])


def test_dropout_ext_repeat_calc():
    """
    Feature: test dropout_ext when repeated calculate
    Description: semi_auto_parallel, dp=2, mp=4
    Expectation: success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((4, 4), (4, 4))
    strategy2 = ((2, 4),)
    net = Net(_w1, strategy1, strategy2)
    phase = compile_net(net, _x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_strategy("DropoutExt-op0", [[2, 4], [], []])
    assert validator.check_node_strategy("DropoutExt-op1", [[2, 4], [], []])
