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

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, mul_weight, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.neg = P.Neg().shard(strategy2)
        self.mul_weight = Parameter(mul_weight, "w1")

    def construct(self, x, b):
        out = self.mul(x, self.mul_weight)
        out = self.neg(out)
        return out, b


_x = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_w1 = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)


def compile_net(net):
    net.set_train()
    _cell_graph_executor.compile(net, _x, _b)
    context.reset_auto_parallel_context()


def test_forward_graph_data_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((16, 1, 1), (16, 1, 1))
    strategy2 = ((16, 1, 1),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_forward_graph_model_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((1, 1, 16), (1, 1, 16))
    strategy2 = ((1, 1, 16),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_forward_graph_hybrid_parallel():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4), (2, 2, 4))
    strategy2 = ((2, 2, 4),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)


def test_forward_graph_auto_parallel():
    """
    Feature: test auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation", device_num=16,
                                      global_rank=0)
    net = Net(_w1)
    compile_net(net)


def test_forward_graph_repeat_calc():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=16, global_rank=0)
    strategy1 = ((2, 2, 4), (2, 2, 4))
    strategy2 = ((1, 2, 2),)
    net = Net(_w1, strategy1, strategy2)
    compile_net(net)
