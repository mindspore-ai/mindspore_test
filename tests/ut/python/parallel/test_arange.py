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

import mindspore.common.dtype as mstype
from mindspore import context, Symbol, Tensor
from mindspore.nn import Cell
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate.gen_ops_prim import Arange, Mul

from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class Net(Cell):
    def __init__(self, strategy_arange=None, strategy_mul=None):
        super(Net, self).__init__()
        self.arange = _get_cache_prim(Arange)()
        if strategy_arange is not None:
            self.arange.shard(strategy_arange)
        self.mul = _get_cache_prim(Mul)()
        if strategy_mul is not None:
            self.mul.shard(strategy_mul)

    def construct(self, x, dtype=None):
        x = self.mul(x, x)
        start = x.shape[0]
        end = x.shape[1]
        step = x.shape[2]
        return self.arange(start, end, step, dtype) * x


def test_arange_dyn_0():
    """
    Feature: test Arange data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    s1 = Symbol(divisor=2)
    x = Tensor(shape=[1, 9, s1], dtype=mstype.float32)
    net = Net()
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs(
        'Arange-0', [1, 9, 'TupleGetItem-1', 'None'])


def test_arange_dyn_1():
    """
    Feature: test Arange parallel
    Description: shape=[1, s1, 1] with ((8,),)
    Expectation: compile success
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=1)
    s1 = Symbol(divisor=3)
    x = Tensor(shape=[1, s1, 1], dtype=mstype.float32)
    strategy_arange = ((8,),)
    net = Net(strategy_arange)
    phase = compile_net(net, x, mstype.float32)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('ScalarSub-0', ['TupleGetItem-1', 1])
    assert validator.check_node_inputs('ScalarDiv-0', ['ScalarSub-0', 8])
    assert validator.check_node_inputs('ScalarMul-0', ['ScalarDiv-0', 1])
    assert validator.check_node_inputs('ScalarAdd-0', [1, 'ScalarMul-0'])
    assert validator.check_node_inputs(
        'ScalarAdd-1', ['ScalarAdd-0', 'ScalarDiv-0'])
    assert validator.check_node_inputs('ScalarCast-0', ['ScalarAdd-0', 43])
    assert validator.check_node_inputs('ScalarCast-1', ['ScalarAdd-1', 43])
    assert validator.check_node_inputs('ScalarCast-2', [1, 43])
    assert validator.check_node_inputs(
        'Arange-0', ['ScalarCast-0', 'ScalarCast-1', 'ScalarCast-2', 43])


def test_arange_dyn_2():
    """
    Feature: test Arange parallel
    Description: shape=[s1, 9, 1] with ((4,),)
    Expectation: compile success
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=7)
    s1 = Symbol(divisor=1)
    x = Tensor(shape=[s1, 9, 1], dtype=mstype.float32)
    strategy_arange = ((4,),)
    net = Net(strategy_arange)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('ScalarSub-0', [9, 'TupleGetItem-1'])
    assert validator.check_node_inputs('ScalarDiv-0', ['ScalarSub-0', 4])
    assert validator.check_node_inputs('ScalarMul-0', ['ScalarDiv-0', 3])
    assert validator.check_node_inputs('ScalarAdd-0', [1, 'ScalarMul-0'])
    assert validator.check_node_inputs(
        'ScalarAdd-1', ['ScalarAdd-0', 'ScalarDiv-0'])
    assert validator.check_node_inputs('ScalarCast-0', ['ScalarAdd-0', 35])
    assert validator.check_node_inputs('ScalarCast-1', ['ScalarAdd-1', 35])
    assert validator.check_node_inputs('ScalarCast-2', [1, 35])
    assert validator.check_node_inputs(
        'Arange-0', ['ScalarCast-0', 'ScalarCast-1', 'ScalarCast-2', 'None'])


def test_arange_dyn_3():
    """
    Feature: test Arange parallel
    Description: shape=[s1, 17, s2] with ((2,),)
    Expectation: compile success
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=3)
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=2)
    x = Tensor(shape=[s1, 17, s2], dtype=mstype.float32)
    strategy_arange = ((2,),)
    net = Net(strategy_arange)
    phase = compile_net(net, x, mstype.float32)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('ScalarSub-0', [17, 'ScalarMul-0'])
    assert validator.check_node_inputs('ScalarDiv-0', ['ScalarSub-0', 2])
    assert validator.check_node_inputs('ScalarMul-0', ['TupleGetItem-3', 2])
    assert validator.check_node_inputs('ScalarMul-1', ['ScalarDiv-0', 0])
    assert validator.check_node_inputs(
        'ScalarAdd-0', ['ScalarMul-0', 'ScalarMul-1'])
    assert validator.check_node_inputs(
        'ScalarAdd-1', ['ScalarAdd-0', 'ScalarDiv-0'])
    assert validator.check_node_inputs('ScalarCast-0', ['ScalarAdd-0', 43])
    assert validator.check_node_inputs('ScalarCast-1', ['ScalarAdd-1', 43])
    assert validator.check_node_inputs('ScalarCast-2', ['TupleGetItem-4', 43])
    assert validator.check_node_inputs(
        'Arange-0', ['ScalarCast-0', 'ScalarCast-1', 'ScalarCast-2', 43])


def test_arange_dyn_4():
    """
    Feature: test Arange parallel
    Description: shape=[s1, s2, s3] with ((2,),)
    Expectation: compile success
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=4)
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=4)
    s3 = Symbol(divisor=2)
    x = Tensor(shape=[s1, s2, s3], dtype=mstype.float32)
    strategy_arange = ((2,),)
    net = Net(strategy_arange)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs(
        'ScalarSub-0', ['TupleGetItem-4', 'ScalarMul-0'])
    assert validator.check_node_inputs('ScalarDiv-0', ['ScalarSub-0', 2])
    assert validator.check_node_inputs('ScalarMul-0', ['TupleGetItem-3', 2])
    assert validator.check_node_inputs('ScalarMul-1', ['ScalarDiv-0', 1])
    assert validator.check_node_inputs(
        'ScalarAdd-0', ['ScalarMul-0', 'ScalarMul-1'])
    assert validator.check_node_inputs(
        'ScalarAdd-1', ['ScalarAdd-0', 'ScalarDiv-0'])
    assert validator.check_node_inputs('ScalarCast-0', ['ScalarAdd-0', 35])
    assert validator.check_node_inputs('ScalarCast-1', ['ScalarAdd-1', 35])
    assert validator.check_node_inputs('ScalarCast-2', ['TupleGetItem-5', 35])
    assert validator.check_node_inputs(
        'Arange-0', ['ScalarCast-0', 'ScalarCast-1', 'ScalarCast-2', 'None'])


def test_arange_dyn_5():
    """
    Feature: test Arange parallel with dynamic shape
    Description: shape=[2, 17, s1] with ((4,),)
    Expectation: compile success
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=5)
    s1 = Symbol(divisor=2)
    x = Tensor(shape=[2, 17, s1], dtype=mstype.float32)
    strategy_arange = ((4,),)
    net = Net(strategy_arange)
    phase = compile_net(net, x, mstype.float32)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('ScalarSub-0', [17, 2])
    assert validator.check_node_inputs('ScalarDiv-0', ['ScalarSub-0', 4])
    assert validator.check_node_inputs('ScalarMul-0', ['ScalarDiv-0', 2])
    assert validator.check_node_inputs('ScalarAdd-0', [2, 'ScalarMul-0'])
    assert validator.check_node_inputs(
        'ScalarAdd-1', ['ScalarAdd-0', 'ScalarDiv-0'])
    assert validator.check_node_inputs('ScalarCast-0', ['ScalarAdd-0', 43])
    assert validator.check_node_inputs('ScalarCast-1', ['ScalarAdd-1', 43])
    assert validator.check_node_inputs('ScalarCast-2', ['TupleGetItem-3', 43])
    assert validator.check_node_inputs(
        'Arange-0', ['ScalarCast-0', 'ScalarCast-1', 'ScalarCast-2', 43])


def test_arange_dyn_6():
    """
    Feature: test Arange parallel
    Description: shape=[8, s1, 16] with ((8,),)
    Expectation: compile success
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=7)
    s1 = Symbol(divisor=32)
    x = Tensor(shape=[8, s1, 16], dtype=mstype.int64)
    strategy_arange = ((4,),)
    strategy_mul = ((1, 8, 1), (1, 8, 1))
    net = Net(strategy_arange, strategy_mul)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('ScalarMul-0', ['TupleGetItem-3', 8])
    assert validator.check_node_inputs('ScalarSub-0', ['ScalarMul-0', 8])
    assert validator.check_node_inputs('ScalarDiv-0', ['ScalarSub-0', 4])
    assert validator.check_node_inputs('ScalarMul-1', ['ScalarDiv-0', 3])
    assert validator.check_node_inputs('ScalarAdd-0', [8, 'ScalarMul-1'])
    assert validator.check_node_inputs(
        'ScalarAdd-1', ['ScalarAdd-0', 'ScalarDiv-0'])
    assert validator.check_node_inputs('ScalarCast-0', ['ScalarAdd-0', 35])
    assert validator.check_node_inputs('ScalarCast-1', ['ScalarAdd-1', 35])
    assert validator.check_node_inputs('ScalarCast-2', [16, 35])
    assert validator.check_node_inputs(
        'Arange-0', ['ScalarCast-0', 'ScalarCast-1', 'ScalarCast-2', 'None'])


def test_arange_static_0():
    """
    Feature: test Arange data parallel
    Description: Constant Folding
    Expectation: compile success
    """
    context.set_auto_parallel_context(
        parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    x = Tensor(shape=[1, 9, 1], dtype=mstype.float32)
    net = Net()
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    cnode_info_dict = validator._get_graph_cnode_info(0)
    assert 'Arange-0' not in cnode_info_dict.keys()
