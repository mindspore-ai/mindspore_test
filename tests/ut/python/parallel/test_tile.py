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
import pytest

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from mindspore.ops import operations as P
from mindspore.parallel.shard import Layout

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, weight, weight2, strategy1=None, strategy2=None, is_parameter=True):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.tile = P.Tile().shard(strategy2)
        if is_parameter:
            self.weight = Parameter(weight, "w1")
        else:
            self.weight = weight
        self.mul2 = P.Mul()
        self.weight2 = Parameter(weight2, "w2")

    def construct(self, x, b):
        out = self.tile(self.weight, (8, 4, 2))
        out = self.mul(x, out)
        out = self.mul2(out, self.weight2)
        return out


class Net2(Cell):
    def __init__(self, weight2, strategy1=None, strategy2=None):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.tile = P.Tile().shard(strategy2)
        self.weight2 = Parameter(weight2, "w2")

    def construct(self, x, b):
        out = self.mul(x, self.weight2)
        out = self.tile(out, (8, 8, 4, 2))
        return out


class Net3(Cell):
    def __init__(self, weight, strategy1=None, strategy2=None, is_parameter=True):
        super().__init__()
        self.mul = P.Mul().shard(strategy1)
        self.tile = P.Tile().shard(strategy2)
        if is_parameter:
            self.weight = Parameter(weight, "w1")
        else:
            self.weight = weight
        self.mul2 = P.Mul()

    def construct(self, x, b):
        out = self.tile(self.weight, (8, 1, 1))
        out = self.mul(x, out)
        return out

class NetLayout(Cell):
    def __init__(self, weight, strategy_tile=None, strategy_mul=None, out_strategy_tile=None, \
                 multiple=(8, 1, 1), is_parameter=True):
        super().__init__()
        if out_strategy_tile is None:
            self.tile = P.Tile().shard(in_strategy=strategy_tile)
        else:
            self.tile = P.Tile().shard(in_strategy=strategy_tile, out_strategy=out_strategy_tile)
        if is_parameter:
            self.weight = Parameter(weight, "w1")
        else:
            self.weight = weight
        self.mul2 = P.Mul().shard(strategy_mul)
        self.multiple = multiple

    def construct(self, x, b):
        out = self.tile(self.weight, self.multiple)
        out = self.mul2(x, out)
        return out

_x = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_x1 = Tensor(np.ones([128, 16, 16]), dtype=ms.float32)
_w1 = Tensor(np.ones([16, 16, 16]), dtype=ms.float32)
_w2 = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)
_w3 = Tensor(np.ones([128, 16, 16]), dtype=ms.float32)
_b = Tensor(np.ones([128, 64, 32]), dtype=ms.float32)


def compile_net(net, x=_b, b=_b):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x, b)
    context.reset_auto_parallel_context()

def test_layout_tile0():
    """
    Feature: test layout tile
    Description: multiple=1 and all split
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    strategy_tile = (layout("dp", "cp", "mp"),)
    strategy_mul2 = (layout("dp", "cp", "mp"), layout("dp", "cp", "mp"))
    multiple = (1, 1, 1)
    net = NetLayout(_b, strategy_tile, strategy_mul2, multiple=multiple, is_parameter=True)
    compile_net(net)

def test_layout_tile1():
    """
    Feature: test layout tile
    Description: correct shard for shape=1 and muliple>1 without out_strategy_tile
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    strategy_tile = (layout("dp", "None", "mp"),)
    strategy_mul2 = (layout("dp", "None", "mp"), layout("dp", "None", "mp"))
    w4 = Tensor(np.ones([128, 1, 32]), dtype=ms.float32)
    multiple = (1, 64, 1)
    net = NetLayout(w4, strategy_tile, strategy_mul2, multiple=multiple, is_parameter=True)
    compile_net(net)

def test_layout_tile2():
    """
    Feature: test layout tile
    Description: correct shard for shape=1 and muliple>1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    strategy_tile = (layout("dp", "None", "mp"),)
    out_strategy_tile = (layout("dp", "cp", "mp"),)
    strategy_mul2 = (layout("dp", "cp", "mp"), layout("dp", "cp", "mp"))
    w4 = Tensor(np.ones([128, 1, 32]), dtype=ms.float32)
    multiple = (1, 64, 1)
    net = NetLayout(w4, strategy_tile, strategy_mul2, out_strategy_tile=out_strategy_tile, \
                    multiple=multiple, is_parameter=True)
    compile_net(net)

def test_layout_tile3():
    """
    Feature: test layout tile
    Description: correct shard for shape>1 and muliple>1
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    strategy_tile = (layout("None", "None", "None"),)
    out_strategy_tile = (layout("dp", "cp", "mp"),)
    strategy_mul2 = (layout("dp", "cp", "mp"), layout("dp", "cp", "mp"))
    w4 = Tensor(np.ones([64, 2, 16]), dtype=ms.float32)
    multiple = (2, 32, 2)
    net = NetLayout(w4, strategy_tile, strategy_mul2, out_strategy_tile=out_strategy_tile, \
                    multiple=multiple, is_parameter=True)
    compile_net(net)


def test_layout_tile4():
    """
    Feature: test layout tile
    Description: wrong shard for shape>1 and muliple>1
    Expectation: raise error
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    strategy_tile = (layout("dp", "cp", "mp"),) # 1th dim of input shouldn't be split
    strategy_mul2 = (layout("dp", "cp", "mp"), layout("dp", "cp", "mp"))
    w4 = Tensor(np.ones([128, 2, 32]), dtype=ms.float32)
    multiple = (1, 32, 1)
    net = NetLayout(w4, strategy_tile, strategy_mul2, multiple=multiple, is_parameter=True)
    with pytest.raises(RuntimeError):
        compile_net(net)

def test_layout_tile5():
    """
    Feature: test layout tile
    Description: correct shard for shape>1 and muliple>1, multiple size is lager than input size
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    strategy_tile = (layout("None", "None"),) # w4 is 2 dims
    out_strategy_tile = (layout("dp", "cp", "mp"),) # multiple is 3 dims, so output is 3 dims
    strategy_mul2 = (layout("dp", "cp", "mp"), layout("dp", "cp", "mp"))
    w4 = Tensor(np.ones([2, 16]), dtype=ms.float32)
    multiple = (128, 32, 2) # this will extend w4 from 2 dims to 3 dims
    net = NetLayout(w4, strategy_tile, strategy_mul2, out_strategy_tile=out_strategy_tile, \
                    multiple=multiple, is_parameter=True)
    compile_net(net)

def test_layout_tile6():
    """
    Feature: test layout tile
    Description: wrong shard for muliple can't be divided by shard num
    Expectation: raise error
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((1, 4, 2), ("dp", "cp", "mp"))
    strategy_tile = (layout("None", "None", "None"),)
    out_strategy_tile = (layout("dp", "cp", "mp"),)
    strategy_mul2 = (layout("dp", "cp", "mp"), layout("dp", "cp", "mp"))
    w4 = Tensor(np.ones([64, 32, 16]), dtype=ms.float32)
    multiple = (2, 2, 2) # the 1th multiple 2 can't be divided by cp4
    net = NetLayout(w4, strategy_tile, strategy_mul2, out_strategy_tile=out_strategy_tile, \
                    multiple=multiple, is_parameter=True)
    with pytest.raises(RuntimeError):
        compile_net(net)

def test_tile_parameter():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 2),)
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=True)
    compile_net(net)


def test_tile_parameter_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 1),)
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=True)
    compile_net(net)


def test_tile_tensor():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 2),)
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=False)
    compile_net(net)


def test_tile_tensor_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((2, 2, 1),)
    net = Net(_w1, _w2, strategy1, strategy2, is_parameter=False)
    compile_net(net)


def test_tile_tensor_no_full_split2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=32, global_rank=0)
    strategy1 = ((4, 4, 1), (4, 4, 1))
    strategy2 = ((4, 4, 1),)
    net = Net3(_w1, strategy1, strategy2)
    compile_net(net, _x1, _b)


def test_tile_output():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((1, 2, 2, 2),)
    net = Net2(_w2, strategy1, strategy2)
    compile_net(net)


def test_tile_output_no_full_split():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = ((1, 2, 1, 2),)
    net = Net2(_w2, strategy1, strategy2)
    compile_net(net)


def test_tile_no_strategy():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((2, 2, 2), (2, 2, 2))
    strategy2 = None
    net = Net2(_w2, strategy1, strategy2)
    compile_net(net)


def test_tile_auto_parallel():
    """
    Feature: test auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation", device_num=8,
                                      global_rank=0)
    net = Net2(_w2)
    compile_net(net)


def test_tile_auto_parallel_2():
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = Net3(_w1)
    compile_net(net, _x1, _b)
