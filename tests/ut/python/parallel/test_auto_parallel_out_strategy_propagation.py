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
import os
import numpy as np
import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell
from mindspore.ops import operations as P

from mindspore.parallel._auto_parallel_context import _set_ops_strategy_json_config


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
    ms.set_algo_parameters(fully_use_devices=True)
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
    ms.reset_auto_parallel_context()
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
    ms.reset_auto_parallel_context()
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
    ms.reset_auto_parallel_context()
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
    ms.reset_auto_parallel_context()
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
    ms.reset_auto_parallel_context()
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
    ms.reset_auto_parallel_context()
    for (k, v) in strategies.items():
        if re.search('Add-op0', k) is not None:
            assert v == [[dp * mp, 1], [dp * mp, 1]]
            break

def test_sharding_strategy_save_and_load1():
    """
    Feature: test strategy can be saved and loaded
    Description: the sharding strategy would be saved or loaded according to the _set_ops_strategy_json_config
    Expectation: when the type is set to SAVE, the config json file requires to be generated; when the type is set to
    LOAD, the strategy requires to be loaded the same as the SAVEd strategy .
    """

    class NetForSaveAndLoad(Cell):
        def __init__(self, mul_weight1, mul_weight2, in_strategy1, in_strategy2):
            super().__init__()
            self.matmul1 = P.MatMul(transpose_b=True).shard(in_strategy=in_strategy1)
            self.matmul2 = P.MatMul(transpose_b=True).shard(in_strategy=in_strategy2)
            self.add1 = P.Add()
            self.add2 = P.Add()
            self.add3 = P.Add()
            self.mul_weight1 = Parameter(mul_weight1, "w1")
            self.mul_weight2 = Parameter(mul_weight2, "w2")

        def construct(self, x, b1, b2, b3):
            out = self.add1(x, b1)
            out = self.matmul1(out, self.mul_weight1)
            out = self.add2(out, b2)
            out = self.matmul2(out, self.mul_weight2)
            out = self.add3(out, b3)
            return out

    def compile_and_get_strategies(in_strategy1, in_strategy2):
        x = Tensor(np.ones([64, 32]), dtype=ms.float32)
        b1 = Tensor(np.ones([64, 32]), dtype=ms.float32)
        w1 = Tensor(np.ones([8, 32]), dtype=ms.float32)
        b2 = Tensor(np.ones([64, 8]), dtype=ms.float32)
        w2 = Tensor(np.ones([16, 8]), dtype=ms.float32)
        b3 = Tensor(np.ones([64, 16]), dtype=ms.float32)

        net = NetForSaveAndLoad(w1, w2, in_strategy1, in_strategy2)
        net.set_train()
        _cell_graph_executor.compile(net, x, b1, b2, b3, phase='train')
        strategies = _cell_graph_executor._get_shard_strategy(net)
        return strategies

    def assert_sharding_strategy(dp1, mp1, dp2, mp2, strategies):
        for (k, v) in strategies.items():
            if re.search('Add-op0', k) is not None:
                assert v == [[dp2, mp2], [dp2, mp2]]
            if re.search('Add-op1', k) is not None:
                assert v == [[dp2, mp2], [dp2, mp2]]
            if re.search('Add-op2', k) is not None:
                assert v == [[dp1, mp1], [dp1, mp1]]
            if re.search('MatMul-op0', k) is not None:
                assert v == [[dp2, mp2], [1, mp2]]
            if re.search('MatMul-op1', k) is not None:
                assert v == [[dp1, mp1], [1, mp1]]

    _dp1 = 4
    _mp1 = 2
    _dp2 = 2
    _mp2 = 4

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    _set_ops_strategy_json_config(type="SAVE", path="./tmp/strategy.json", mode="all")

    _in_strategy1 = ((_dp1, _mp1), (1, _mp1))
    _in_strategy2 = ((_dp2, _mp2), (1, _mp2))
    if os.path.exists("./tmp/strategy.json"):
        os.remove("./tmp/strategy.json")
    _strategies = compile_and_get_strategies(_in_strategy1, _in_strategy2)
    _strategies = compile_and_get_strategies(_in_strategy1, _in_strategy2)
    assert os.path.exists("./tmp/strategy.json")
    assert_sharding_strategy(_dp1, _mp1, _dp2, _mp2, _strategies)
    ms.reset_auto_parallel_context()

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    _set_ops_strategy_json_config(type="LOAD", path="./tmp/strategy.json", mode="all")
    _in_strategy1 = None
    _in_strategy2 = None
    _strategies = compile_and_get_strategies(_in_strategy1, _in_strategy2)
    assert_sharding_strategy(_dp1, _mp1, _dp2, _mp2, _strategies)
    ms.reset_auto_parallel_context()
    os.remove("./tmp/strategy.json")


def test_sharding_strategy_save_and_load2():
    """
    Feature: test strategy can be saved and loaded when ops has out_strategy
    Description: the sharding strategy would be saved or loaded according to the _set_ops_strategy_json_config
    Expectation: when the type is set to SAVE, the config json file requires to be generated; when the type is set to
    LOAD, the strategy requires to be loaded the same as the SAVEd strategy .
    """

    class NetForSaveAndLoad(Cell):
        def __init__(self, mul_weight1, mul_weight2, in_strategy1, in_strategy2, out_strategy1, out_strategy2):
            super().__init__()
            self.matmul1 = P.MatMul(transpose_b=True).shard(in_strategy=in_strategy1, out_strategy=out_strategy1)
            self.matmul2 = P.MatMul(transpose_b=True).shard(in_strategy=in_strategy2, out_strategy=out_strategy2)
            self.add1 = P.Add()
            self.add2 = P.Add()
            self.add3 = P.Add()
            self.mul_weight1 = Parameter(mul_weight1, "w1")
            self.mul_weight2 = Parameter(mul_weight2, "w2")

        def construct(self, x, b1, b2, b3):
            out = self.add1(x, b1)
            out = self.matmul1(out, self.mul_weight1)
            out = self.add2(out, b2)
            out = self.matmul2(out, self.mul_weight2)
            out = self.add3(out, b3)
            return out

    def compile_and_get_strategies(in_strategy1, in_strategy2, out_strategy1, out_strategy2):
        x = Tensor(np.ones([64, 32]), dtype=ms.float32)
        b1 = Tensor(np.ones([64, 32]), dtype=ms.float32)
        w1 = Tensor(np.ones([8, 32]), dtype=ms.float32)
        b2 = Tensor(np.ones([64, 8]), dtype=ms.float32)
        w2 = Tensor(np.ones([16, 8]), dtype=ms.float32)
        b3 = Tensor(np.ones([64, 16]), dtype=ms.float32)

        net = NetForSaveAndLoad(w1, w2, in_strategy1, in_strategy2, out_strategy1, out_strategy2)
        net.set_train()
        _cell_graph_executor.compile(net, x, b1, b2, b3, phase='train')
        strategies = _cell_graph_executor._get_shard_strategy(net)
        return strategies

    def assert_sharding_strategy(dp1, mp1, dp2, mp2, strategies):
        for (k, v) in strategies.items():
            if re.search('Add-op0', k) is not None:
                assert v == [[8, 1], [8, 1]]
            if re.search('Add-op1', k) is not None:
                assert v == [[dp2, mp2], [dp2, mp2]]
            if re.search('Add-op2', k) is not None:
                assert v == [[dp1, mp1], [dp1, mp1]]
            if re.search('MatMul-op0', k) is not None:
                assert v == [[dp2, mp2], [1, mp2]]
            if re.search('MatMul-op1', k) is not None:
                assert v == [[dp1, mp1], [1, mp1]]

    _dp1 = 4
    _mp1 = 2
    _dp2 = 2
    _mp2 = 4

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    ms.set_algo_parameters(fully_use_devices=True)
    _set_ops_strategy_json_config(type="SAVE", path="./tmp/strategy.json", mode="all")

    _in_strategy1 = ((_dp1, _mp1), (1, _mp1))
    _out_strategy1 = ((_dp1 * _mp1, 1),)
    _in_strategy2 = ((_dp2, _mp2), (1, _mp2))
    _out_strategy2 = ((_dp2 * _mp2, 1),)
    if os.path.exists("./tmp/strategy.json"):
        os.remove("./tmp/strategy.json")
    _strategies = compile_and_get_strategies(_in_strategy1, _in_strategy2, _out_strategy1, _out_strategy2)
    _strategies = compile_and_get_strategies(_in_strategy1, _in_strategy2, _out_strategy1, _out_strategy2)
    assert os.path.exists("./tmp/strategy.json")
    assert_sharding_strategy(_dp1, _mp1, _dp2, _mp2, _strategies)
    ms.reset_auto_parallel_context()

    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    _set_ops_strategy_json_config(type="LOAD", path="./tmp/strategy.json", mode="all")
    _in_strategy1 = None
    _out_strategy1 = None
    _in_strategy2 = None
    _out_strategy2 = None
    _strategies = compile_and_get_strategies(_in_strategy1, _in_strategy2, _out_strategy1, _out_strategy2)
    assert_sharding_strategy(_dp1, _mp1, _dp2, _mp2, _strategies)
    ms.reset_auto_parallel_context()
    os.remove("./tmp/strategy.json")
