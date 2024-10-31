# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
import numpy as np
import mindspore.nn as nn
import mindspore.dataset as ds
from mindspore import context, Tensor
from mindspore.train import Model
from mindspore.common.api import _cell_graph_executor
from mindspore.nn import TrainOneStepCell, WithLossCell, Momentum
from mindspore.communication.management import init, create_group, destroy_group, get_group_size, get_rank, \
    get_local_rank, get_world_rank_from_group_rank, get_group_rank_from_world_rank
from tests.mark_utils import arg_mark

os.environ["MS_SIMULATION_LEVEL"] = "0"
context.set_context(mode=context.GRAPH_MODE)


class DenseNet(nn.Cell):
    def __init__(self, has_bias=True, activation='relu'):
        super(DenseNet, self).__init__()
        self.fc1 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc2 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)
        self.fc3 = nn.Dense(128, 128, has_bias=has_bias, activation=activation)

    def construct(self, x):
        q = self.fc1(x)
        k = self.fc2(q)
        v = self.fc3(k)
        return v


input_ = Tensor(np.ones([32, 128]).astype(np.float32) * 0.01)
label_ = Tensor(np.zeros([32, 128]).astype(np.float32))


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_run_graph_kbk():
    """
    Feature: simulation level.
    Description: run graph when set simulation level 1.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "1"
    os.environ["OMPI_COMMAND"] = "1"
    os.environ["PMIX_RANK"] = "1"
    context.set_context(jit_level='O0')
    init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = DenseNet()
    net.fc1.matmul.shard(((4, 1), (8, 1)))
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    net = WithLossCell(net, loss_fn)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net(input_, label_)
    context.reset_auto_parallel_context()
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_get_rank_id_env():
    """
    Feature: simulation level.
    Description: get rank id default when set simulation level 0.
    Expectation: return env rank id.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "8"
    os.environ["RANK_ID"] = "7"
    init()
    ret = get_rank()
    assert ret == 7
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_get_local_rank_id():
    """
    Feature: simulation level.
    Description: get local rank id when set simulation level 0.
    Expectation: return local rank id.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "9"
    init()
    ret = get_local_rank()
    assert ret == 1
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_create_group():
    """
    Feature: simulation level.
    Description: create group when set simulation level 0.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "7"
    init()
    group = "g-01234567"
    rank_ids = [i for i in range(8)]
    create_group(group, rank_ids)
    ret = get_group_size()
    assert ret == 32
    ret = get_group_size(group)
    assert ret == 8
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
def test_destroy_group():
    """
    Feature: simulation level.
    Description: destroy group when set simulation level 0.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "7"
    init()
    group = "g8-01234567"
    rank_ids = [i for i in range(8)]
    create_group(group, rank_ids)
    destroy_group(group)
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_get_world_rank_from_group_rank():
    """
    Feature: simulation level.
    Description: get world rank from group rank when set simulation level 0.
    Expectation: return world rank.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "9"
    init()
    group = "g-to-w-8+01234567"
    rank_ids = [i + 8 for i in range(8)]
    create_group(group, rank_ids)
    ret = get_world_rank_from_group_rank(group, 3)
    assert ret == 11
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_get_group_rank_from_world_rank():
    """
    Feature: simulation level.
    Description: get local rank id default when set simulation level 0.
    Expectation: return local rank id.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "9"
    init()
    group = "w-to-g-8+01234567"
    rank_ids = [i + 8 for i in range(8)]
    create_group(group, rank_ids)
    ret = get_group_rank_from_world_rank(12, group)
    assert ret == 4
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="unessential")
def test_simulation_graph():
    """
    Feature: simulation level.
    Description: compile graph when set simulation level 0.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "1"
    init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = DenseNet()
    net.fc1.matmul.shard(((4, 1), (8, 1)))
    _cell_graph_executor.compile(net, input_)
    context.reset_auto_parallel_context()
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_run_graph():
    """
    Feature: simulation level.
    Description: run graph when set simulation level 1.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "1"
    init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = DenseNet()
    net.fc1.matmul.shard(((4, 1), (8, 1)))
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    net = WithLossCell(net, loss_fn)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    train_net(input_, label_)
    context.reset_auto_parallel_context()
    os.environ["MS_SIMULATION_LEVEL"] = ""


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_build_model_with_dataset():
    """
    Feature: simulation level.
    Description: run graph when set simulation level 1.
    Expectation: no exception.
    """
    os.environ["MS_SIMULATION_LEVEL"] = "1"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "1"
    init()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    net = DenseNet()
    net.fc1.matmul.shard(((4, 1), (8, 1)))
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits()
    data_list = []
    for _ in range(8):
        data_list.append((np.ones([32, 128]).astype(np.float32), np.zeros([32, 128]).astype(np.float32)))
    dataset = ds.GeneratorDataset(data_list, ["input", "label"])
    model = Model(net, loss_fn, optimizer)
    model.build(dataset)
    context.reset_auto_parallel_context()
    os.environ["MS_SIMULATION_LEVEL"] = ""
