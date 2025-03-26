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
from mindspore import context, Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.parallel.shard import Layout
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import Cell, TrainOneStepCell, Momentum
from parallel.utils.utils import ParallelValidator
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode, remove_files, find_ir_file_path, \
    check_node_attrs_pair


def setup_function():
    keyword = 'enable_alltoall'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


def teardown_function():
    keyword = 'enable_alltoall'
    base_dir = './test_auto_parallel'
    remove_files(keyword, base_dir)


def compile_net(net, _x1, _b1):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net.set_inputs(_x1, _b1)

    # set auto_parallel
    init_hccl(global_rank=0, device_num=8)
    parallel_config = {"parallel_mode": "semi_auto", "gradients_mean": True,
                       "dataset_strategy": "full_batch"}
    train_net = set_parallel_mode(train_net, parallel_config)

    # compile net
    phase, _ = _cell_graph_executor.compile(train_net, _x1, _b1)
    return phase, train_net


def test_layout():
    """
    Feature: config layout for dynamic shape
    Description: no redistribution
    Expectation: compile success
    """
    graph_path = "./test_auto_parallel/test_enable_alltoall_graphs"
    context.set_context(save_graphs=True, save_graphs_path=graph_path)

    class DynamicMulNet(Cell):
        def __init__(self, strategy1):
            super().__init__()
            self.mul = P.Mul().shard(strategy1)
            layout = Layout((8, 1, 1), ("dp", "mp", "xp"))
            layout1 = (layout("dp", "mp", "xp"),)

            self.gelu = P.GeLU().shard(layout1)
            self.w = Parameter(Tensor(np.ones([1]), dtype=ms.float32), "w2")

        def construct(self, x, y):
            out = self.mul(x, self.w)
            out = self.gelu(out)
            return out

    strategy1 = ((8, 1, 1), (1,))
    net = DynamicMulNet(strategy1)

    x = Tensor(shape=[16, None, 1], dtype=ms.float32)
    y = Tensor(shape=[None, None, None], dtype=ms.float32)

    net.set_inputs(x, y)
    phase, _ = compile_net(net, x, y)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('GeLU-0', ['Mul-0'])

    # enable_alltoall is true
    validate_ir = find_ir_file_path(graph_path, "step_parallel_end")
    check_pairs = {"_MirrorOperator": {"mean_flag: Bool(1)": 1}}
    check_node_attrs_pair(validate_ir, check_pairs)
