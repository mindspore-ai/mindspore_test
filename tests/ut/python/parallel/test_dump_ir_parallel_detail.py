# Copyright 2024-2025 Huawei Technologies Co., Ltd
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

import os
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore import context
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore.parallel.shard import Layout
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.utils.utils import ParallelValidator, compile_net
from parallel.auto_parallel_interface._utils import find_ir_file_path, check_node_pairs_num


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


grad_all = C.GradOperation(get_all=True)


class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y):
        predict = self.network(x, y)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y):
        return grad_all(self.network)(x, y)


class Net_semi(nn.Cell):
    def __init__(self, shape, axis=0, strategy1=None, strategy2=None, batch_dims=0, gather_out_strategy=None,
                 weight_shape=None):
        super().__init__()
        self.gatherv2 = P.Gather(batch_dims=batch_dims).shard(strategy1, out_strategy=gather_out_strategy)
        self.mul = P.Mul().shard(strategy2)
        self.relu = P.ReLU()
        if strategy2:
            self.relu.shard((strategy2[0],))
        self.index = None
        if shape:
            self.index = Tensor(np.ones(shape), dtype=ms.int32)
        else:
            self.index = Tensor(1, dtype=ms.int32)
        self.axis = axis
        self.weight_shape = weight_shape
        if self.weight_shape is not None:
            self.weight = Parameter(Tensor(np.ones(self.weight_shape), dtype=ms.float32), name="weight")

    def construct(self, x, y):
        out = self.gatherv2(x, self.index, self.axis)
        out = self.relu(out)
        out = self.mul(out, y)
        if self.weight_shape is not None:
            out = self.mul(out, self.weight)
        return out


class Net_auto(nn.Cell):
    def __init__(self, shape, axis=0, strategy1=None, layout=None, batch_dims=0, gather_out_strategy=None,
                 weight_shape=None):
        super().__init__()
        self.gatherv2 = P.Gather(batch_dims=batch_dims).shard(strategy1, out_strategy=gather_out_strategy)
        self.mul = P.Mul().shard(layout)
        self.relu1 = P.ReLU()
        self.relu2 = P.ReLU().shard(strategy1)
        if layout:
            self.relu1.shard(layout)
        self.index = None
        if shape:
            self.index = Tensor(np.ones(shape), dtype=ms.int32)
        else:
            self.index = Tensor(1, dtype=ms.int32)
        self.axis = axis
        self.weight_shape = weight_shape
        if self.weight_shape is not None:
            self.weight = Parameter(Tensor(np.ones(self.weight_shape), dtype=ms.float32), name="weight")

    def construct(self, x, y):
        out = self.relu1(x)
        out = out - x
        out = self.relu2(out)
        out = out - y
        return out


def compile_graph(net, x, y):
    os.environ['MS_DEV_DUMP_IR_PARALLEL_DETAIL'] = '1'
    net.set_train()
    phase = compile_net(net, x, y)
    os.environ['MS_DEV_DUMP_IR_PARALLEL_DETAIL'] = ''
    return phase


def test_dump_ir_parallel_detail_semi():
    """
    Feature: print dump IR of tensor_map & device_matrix
    Description: net with strategy in semi auto parallel, print Dump IR parallel detail.
    Expectation: compile done without error.
    """
    context.set_context(save_graphs=True, save_graphs_path="./test_dump_ir_parallel_detail/semi_auto_graphs")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 2), (4,))
    strategy2 = ((4, 2), (4, 2))
    net = GradWrap(NetWithLoss(Net_semi(shape=(8,), axis=0, strategy1=strategy1, strategy2=strategy2)))
    x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([8, 64]), dtype=ms.float32)
    phase = compile_graph(net, x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs("ReLU-0", ["Gather-0"])
    assert validator.check_node_attrs("Gather-0", {"inputs_tensor_map": "((1, 0), (2))", "device_matrix": "(4, 1, 2)",
                                                   "outputs_tensor_map": "((2, 0))"}, 0)
    assert validator.check_node_attrs("ReLU-0", {"inputs_tensor_map": "((1, 0))", "device_matrix": "(4, 2)",
                                                 "outputs_tensor_map": "((1, 0))"}, 0)
    assert validator.check_node_attrs("Mul-0", {"inputs_tensor_map": "((1, 0), (1, 0))", "device_matrix": "(4, 2)",
                                                "outputs_tensor_map": "((1, 0))"}, 0)


def test_dump_ir_parallel_detail_auto():
    """
    Feature: print dump IR of tensor_map & device_matrix
    Description: net with strategy in auto parallel & sharding_propagation  model, print Dump IR parallel detail.
    Expectation: compile done without error.
    """
    graph_path = "./test_dump_ir_parallel_detail/sharding_propagation_graphs"
    context.set_context(save_graphs=True, save_graphs_path=graph_path)
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=8, global_rank=0,
                                      search_mode="sharding_propagation")
    strategy1 = ((2, 4),)
    layout = Layout((8, 1), ("dp", "mp"))
    layout1 = (layout("dp", "mp"),)
    net = GradWrap(NetWithLoss(Net_auto(shape=(), axis=0, strategy1=strategy1, layout=layout1)))
    x = Tensor(np.ones([32, 64]), dtype=ms.float32)
    y = Tensor(np.ones([64]), dtype=ms.float32)
    compile_graph(net, x, y)

    validate_ir = find_ir_file_path(graph_path, "validate")
    check_pair = {"out_layout": "4"}
    check_node_pairs_num(validate_ir, check_pair)
