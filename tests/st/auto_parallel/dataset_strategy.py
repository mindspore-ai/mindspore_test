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
# ============================================================================

import numpy as np
import mindspore as ms
from mindspore import nn, context
from mindspore.context import ParallelMode
from mindspore.communication.management import init
from mindspore.nn.utils import no_init_parameters
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.train import Model
from .model_parallel import FakeData, FakeDataInitMode, ParallelNetwork
from .model_parallel import allclose_nparray, CustomLossMonitor


shard_strategy = ((1, 1), (1, 2))
ms.set_seed(1)


def train_using_auto_parallel_cell(dataset_strategy):
    ms.set_seed(1)
    parallel_ir_path = "data_strategy_ir/parallel_ir_" + str(dataset_strategy)
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path=parallel_ir_path)
    with no_init_parameters():
        net = ParallelNetwork(strategy=shard_strategy)
        optimizer = nn.Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
    parallel_net.dataset_strategy(dataset_strategy)
    parallel_dataset = FakeData(size=32, batch_size=8, image_size=(
        4, 4), num_classes=10, fakedata_mode=FakeDataInitMode.UniqueInit)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor = CustomLossMonitor(per_print_times=1)
    model = Model(network=parallel_net, loss_fn=loss_fn, optimizer=optimizer)
    print("the loss when training using AutoParallel(cell) is :")
    model.train(epoch=2, train_dataset=parallel_dataset,
                dataset_sink_mode=False, callbacks=[loss_monitor])
    return loss_monitor.loss_history


def train_using_set_auto_parallel_context(dataset_strategy):
    ms.set_seed(1)
    dataset = FakeData(size=32, batch_size=8, image_size=(
        4, 4), num_classes=10, fakedata_mode=FakeDataInitMode.UniqueInit)
    ir_path = "data_strategy_ir/context_ir_" + str(dataset_strategy)
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path=ir_path)
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL, dataset_strategy=dataset_strategy)
    net = ParallelNetwork(strategy=shard_strategy)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = nn.Momentum(net.trainable_params(),
                            learning_rate=0.1, momentum=0.9)
    loss_monitor = CustomLossMonitor(per_print_times=1)
    model = Model(network=net, loss_fn=loss_fn, optimizer=optimizer)
    print("the loss when training using set_auto_parallel_context is :")
    model.train(epoch=2, train_dataset=dataset,
                dataset_sink_mode=False, callbacks=[loss_monitor])
    return loss_monitor.loss_history


def test_dataset_strategy_data_parallel():
    '''
    Feature: AutoParallel(cell).dataset_strategy(config)
    Description: Test interface AutoParallel(cell).dataset_strategy("data_parallel")
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    '''
    init(backend_name='hccl')
    dataset_strategy = "data_parallel"
    baseline_loss = train_using_set_auto_parallel_context(
        dataset_strategy)
    context.reset_auto_parallel_context()
    parallel_loss = train_using_auto_parallel_cell(
        dataset_strategy)
    allclose_nparray(np.array(parallel_loss),
                     np.array(baseline_loss), 0.001, 0.001)


def test_dataset_strategy_full_batch():
    '''
    Feature: AutoParallel(cell).dataset_strategy(config)
    Description: Test interface AutoParallel(cell).dataset_strategy("full_batch")
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    '''
    init(backend_name='hccl')
    dataset_strategy = "full_batch"
    baseline_loss = train_using_set_auto_parallel_context(
        dataset_strategy)
    context.reset_auto_parallel_context()
    parallel_loss = train_using_auto_parallel_cell(
        dataset_strategy)
    allclose_nparray(np.array(parallel_loss),
                     np.array(baseline_loss), 0.001, 0.001)


def test_dataset_strategy_using_tuple():
    '''
    Feature: AutoParallel(cell).dataset_strategy(config)
    Description: Test interface AutoParallel(cell).dataset_strategy(((2,1,1),(2,1)))
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    '''
    init(backend_name='hccl')
    dataset_strategy = ((2, 1, 1), (2, 1))
    baseline_loss = train_using_set_auto_parallel_context(
        dataset_strategy)
    context.reset_auto_parallel_context()
    parallel_loss = train_using_auto_parallel_cell(
        dataset_strategy)
    allclose_nparray(np.array(parallel_loss),
                     np.array(baseline_loss), 0.001, 0.001)
