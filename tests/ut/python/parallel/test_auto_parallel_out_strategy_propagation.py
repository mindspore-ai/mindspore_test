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

import re
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P


def test_out_strategy_propagate1():
    """
    Feature: test out_strategy can be propagated to next operator
    Description: when out_strategy is not set, strategy of next operator can be propagated
    Expectation: strategy of next operator is correct
    """

    class NetForOutStrategy(Cell):
        def __init__(self, mul_weight, in_strategy, out_strategy):
            super().__init__()
            self.matmul = P.MatMul(transpose_b=True).shard(in_strategy=in_strategy, out_strategy=out_strategy)
            self.add = P.Add()
            self.relu = P.ReLU()
            self.mul_weight = Parameter(mul_weight, "w")

        def construct(self, x, b):
            out = self.matmul(x, self.mul_weight)
            out = self.add(out, b)
            out = self.relu(out)
            return out

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    dp = 4
    mp = 2
    _x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    _w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    _b = Tensor(np.ones([64, 8]), dtype=ms.float32)
    in_strategy = ((dp, mp), (1, mp))
    net = NetForOutStrategy(_w, in_strategy, None)
    net.set_train()
    _cell_graph_executor.compile(net, _x, _b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context.reset_auto_parallel_context()
    for (k, v) in strategies.items():
        if re.search('Add-op0', k) is not None:
            assert v == [[dp, mp], [dp, mp]]
            break


def test_out_strategy_propagate2():
    """
    Feature: test out_strategy can be propagated to next operator
    Description: when out_strategy is set, strategy of next operator can be propagated
    Expectation: strategy of next operator is correct
    """

    class NetForOutStrategy(Cell):
        def __init__(self, mul_weight, in_strategy, out_strategy):
            super().__init__()
            self.matmul = P.MatMul(transpose_b=True).shard(in_strategy=in_strategy, out_strategy=out_strategy)
            self.relu = P.ReLU()
            self.add = P.Add()
            self.mul_weight = Parameter(mul_weight, "w")

        def construct(self, x, b):
            out = self.relu(x)
            out = self.matmul(out, self.mul_weight)
            out = self.add(out, b)
            return out

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    dp = 4
    mp = 2
    _x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    _w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    _b = Tensor(np.ones([64, 8]), dtype=ms.float32)
    in_strategy = ((dp, mp), (1, mp))
    out_strategy = ((dp * mp, 1),)
    net = NetForOutStrategy(_w, in_strategy, out_strategy)
    net.set_train()
    _cell_graph_executor.compile(net, _x, _b, phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context.reset_auto_parallel_context()
    for (k, v) in strategies.items():
        if re.search('Add-op0', k) is not None:
            assert v == [[dp * mp, 1], [dp * mp, 1]]
            break


def test_out_strategy_propagate3():
    """
    Feature: test out_strategy can be propagated to next operator
    Description: when out_strategy is set, strategy of next operator can be propagated
    Expectation: strategy of next operator is correct
    """

    class NetForOutStrategy(Cell):
        def __init__(self, mul_weight, in_strategy, out_strategy):
            super().__init__()
            self.reshape = P.Reshape()
            self.matmul = P.MatMul(transpose_b=True).shard(in_strategy=in_strategy, out_strategy=out_strategy)
            self.relu = P.ReLU()
            self.mul_weight = Parameter(mul_weight, "w")

        def construct(self, x, reshape):
            out = self.reshape(x, reshape)
            out = self.matmul(out, self.mul_weight)
            out = self.relu(out)
            return out

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    dp = 4
    mp = 2
    _x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    _w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    in_strategy = ((dp, mp), (1, mp))
    out_strategy = ((dp * mp, 1),)
    net = NetForOutStrategy(_w, in_strategy, out_strategy)
    net.set_train()
    _cell_graph_executor.compile(net, _x, (-1, 32), phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context.reset_auto_parallel_context()
    for (k, v) in strategies.items():
        if re.search('ReLU-op', k) is not None:
            assert v == [[dp * mp, 1]]
            break


def test_out_strategy_propagate4():
    """
    Feature: test out_strategy can be propagated to next operator
    Description: after reshape the shape is 4 which cannot be divided into 8, so reshape strategy is (4,2)
    Expectation: strategy of next operator is correct
    """

    class NetForOutStrategy(Cell):
        def __init__(self, mul_weight, in_strategy, out_strategy):
            super().__init__()
            self.reshape = P.Reshape()
            self.matmul = P.MatMul(transpose_b=True).shard(in_strategy=in_strategy, out_strategy=out_strategy)
            self.relu = P.ReLU()
            self.mul_weight = Parameter(mul_weight, "w")

        def construct(self, x, reshape):
            out = self.matmul(x, self.mul_weight)
            out = self.reshape(out, reshape)
            out = self.relu(out)
            return out

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    dp = 4
    mp = 2
    _x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    _w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    in_strategy = ((dp, mp), (1, mp))
    out_strategy = ((dp * mp, 1),)
    net = NetForOutStrategy(_w, in_strategy, out_strategy)
    net.set_train()
    _cell_graph_executor.compile(net, _x, (4, -1), phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context.reset_auto_parallel_context()
    for (k, v) in strategies.items():
        if re.search('ReLU-op', k) is not None:
            assert v == [[4, 2]]
            break


def test_out_strategy_propagate5():
    """
    Feature: test out_strategy can be propagated to next operator
    Description: after reshape the shape is 16 which can be divided into 8, so reshape strategy is (8,1)
    Expectation: strategy of next operator is correct
    """

    class NetForOutStrategy(Cell):
        def __init__(self, mul_weight, in_strategy, out_strategy):
            super().__init__()
            self.reshape = P.Reshape()
            self.matmul = P.MatMul(transpose_b=True).shard(in_strategy=in_strategy, out_strategy=out_strategy)
            self.relu = P.ReLU()
            self.mul_weight = Parameter(mul_weight, "w")

        def construct(self, x, reshape):
            out = self.matmul(x, self.mul_weight)
            out = self.reshape(out, reshape)
            out = self.relu(out)
            return out

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    dp = 4
    mp = 2
    _x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    _w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    in_strategy = ((dp, mp), (1, mp))
    out_strategy = ((dp * mp, 1),)
    net = NetForOutStrategy(_w, in_strategy, out_strategy)
    net.set_train()
    _cell_graph_executor.compile(net, _x, (16, -1), phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context.reset_auto_parallel_context()
    for (k, v) in strategies.items():
        if re.search('ReLU-op', k) is not None:
            assert v == [[dp * mp, 1]]
            break



def test_out_strategy_propagate6():
    """
    Feature: test out_strategy can be propagated to next operator
    Description: when out_strategy is set, strategy of next operator can be propagated
    Expectation: strategy of next operator is correct
    """

    class NetForOutStrategy(Cell):
        def __init__(self, mul_weight, in_strategy, out_strategy):
            super().__init__()
            self.reshape = P.Reshape()
            self.matmul = P.MatMul(transpose_b=True).shard(in_strategy=in_strategy, out_strategy=out_strategy)
            self.add = P.Add()
            self.relu = P.ReLU()
            self.mul_weight = Parameter(mul_weight, "w")

        def construct(self, x, b, reshape):
            out = self.matmul(x, self.mul_weight)
            out = self.reshape(out, reshape)
            out = self.add(out, b)
            out = self.relu(out)
            return out

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    dp = 4
    mp = 2
    _x = Tensor(np.ones([64, 32]), dtype=ms.float32)
    _w = Tensor(np.ones([8, 32]), dtype=ms.float32)
    _b = Tensor(np.ones([32, 16]), dtype=ms.float32)
    in_strategy = ((dp, mp), (1, mp))
    out_strategy = ((dp * mp, 1),)
    net = NetForOutStrategy(_w, in_strategy, out_strategy)
    net.set_train()
    _cell_graph_executor.compile(net, _x, _b, (-1, 16), phase='train')
    strategies = _cell_graph_executor._get_shard_strategy(net)
    context.reset_auto_parallel_context()
    for (k, v) in strategies.items():
        if re.search('Add-op0', k) is not None:
            assert v == [[dp * mp, 1], [dp * mp, 1]]
            break
