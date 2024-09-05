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
# ============================================================================
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore import Symbol
from mindspore.train import Model
from mindspore.nn.wrap.cell_wrapper import PipelineCell
from mindspore.ops import operations as P
from mindspore import lazy_inline
from .test_pipeline_split import DatasetLenet, MatMulCell


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None, dtype=ms.float32):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = MatMulCell(strategy1, strategy2, param, dtype)
            cell.pipeline_stage = i
            if i == 1:
                cell.param.requires_grad = False
                cell.param1.requires_grad = False
            self.block.append(cell)

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        return x


class PipelineSplit(nn.Cell):
    @lazy_inline
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.cell = Net(strategy1, strategy2, dtype=dtype)

    def construct(self, x, label):
        x = self.cell(x)
        return x

class PipelineSplitWithScalarLoss(nn.Cell):
    @lazy_inline
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.cell = Net(strategy1, strategy2, dtype=dtype)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)
        self.loss = P.ReduceSum()

    def construct(self, x, label):
        x = self.cell(x)
        x = self.loss(x)
        return x


def test_pipeline_split_stage0():
    '''
    Feature: pipeline + grad_freeze + stage0
    Description: In pipeline mode, stage1's param's requires_grad = False, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage1():
    '''
    Feature: pipeline + grad_freeze + stage1
    Description: In pipeline mode, stage1's param's requires_grad = False, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_dynamic_shape_pipeline_split_stage0():
    '''
    Feature: pipeline + dynamic_shape + grad_freeze + stage0
    Description: In pipeline mode, stage1's param's requires_grad = False, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplitWithScalarLoss(strategy1, strategy2), 4)
    s1 = Symbol(divisor=4)
    dynamic_data = Tensor(shape=[s1, None], dtype=ms.float32)
    dynamic_label = Tensor(shape=[s1, None], dtype=ms.float32)
    net.set_inputs(dynamic_data, dynamic_label)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_dynamic_shape_pipeline_split_stage1():
    '''
    Feature: pipeline + dynamic_shape + grad_freeze + stage1
    Description: In pipeline mode, stage1's param's requires_grad = False, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplitWithScalarLoss(strategy1, strategy2), 4)
    s1 = Symbol(divisor=4)
    dynamic_data = Tensor(shape=[s1, None], dtype=ms.float32)
    dynamic_label = Tensor(shape=[s1, None], dtype=ms.float32)
    net.set_inputs(dynamic_data, dynamic_label)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)


def test_pipeline_split_stage1_save_stra():
    '''
    Feature: pipeline + grad_freeze + stage1 + opt_shard + save_strategy
    Description: In pipeline mode, stage1's param's requires_grad = False, expected success
    Expectation: success
    '''
    context.set_auto_parallel_context(device_num=32, global_rank=16, pipeline_stages=2, enable_parallel_optimizer=True,
                                      parallel_optimizer_config={"parallel_optimizer_threshold": 1},
                                      strategy_ckpt_config={"save_file": "./strategy_freeze_stage1.ckpt",
                                                            "only_trainable_params": False})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    net = PipelineCell(PipelineSplit(strategy1, strategy2), 4)
    params = net.trainable_params()
    dataset = DatasetLenet(data, label, 3)
    optimizer = nn.Lamb(params, learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    stra = ms.build_searched_strategy("./strategy_freeze_stage1.ckpt")
    assert "cell.block.1.param" in stra
