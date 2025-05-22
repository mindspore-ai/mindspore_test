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


def train_auto_parallel_using_autoparallel_cell(search_mode, strategy=None):
    ms.set_seed(1)
    dataset = FakeData(size=32, batch_size=8, image_size=(
        4, 4), num_classes=10, fakedata_mode=FakeDataInitMode.UniqueInit)
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')
    with no_init_parameters():
        net = ParallelNetwork(strategy=strategy)
        optimizer = nn.Momentum(net.trainable_params(),
                                learning_rate=0.1, momentum=0.9)
    parallel_net = AutoParallel(net, parallel_mode=search_mode)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor = CustomLossMonitor(per_print_times=1)
    model = Model(network=parallel_net, loss_fn=loss_fn, optimizer=optimizer)
    model.train(epoch=2, train_dataset=dataset,
                dataset_sink_mode=False, callbacks=[loss_monitor])
    return loss_monitor.loss_history


def train_auto_parallel_baseline(search_mode, strategy=None):
    ms.set_seed(1)
    dataset = FakeData(size=32, batch_size=8, image_size=(
        4, 4), num_classes=10, fakedata_mode=FakeDataInitMode.UniqueInit)
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend')
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.AUTO_PARALLEL, search_mode=search_mode)
    net = ParallelNetwork(strategy=strategy)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = nn.Momentum(net.trainable_params(),
                            learning_rate=0.1, momentum=0.9)
    loss_monitor = CustomLossMonitor(per_print_times=1)
    model = Model(network=net, loss_fn=loss_fn, optimizer=optimizer)
    model.train(epoch=2, train_dataset=dataset,
                dataset_sink_mode=False, callbacks=[loss_monitor])
    return loss_monitor.loss_history


def test_auto_parallel_sharding_propagation():
    """
    Feature: AutoParallel(cell), parallel_mode is "sharding_propagation"
    Description: Test AutoParallel(net, parallel_mode = "sharding_propagation")
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    init(backend_name='hccl')
    strategy = ((1, 1), (1, 2))
    parallel_mode = "sharding_propagation"
    context_loss = train_auto_parallel_baseline(parallel_mode, strategy)
    context.reset_auto_parallel_context()
    parallel_loss = train_auto_parallel_using_autoparallel_cell(
        parallel_mode, strategy)
    allclose_nparray(np.array(parallel_loss),
                     np.array(context_loss), 0.001, 0.001)


def test_auto_parallel_recursive_programming():
    """
    Feature: AutoParallel(cell), parallel_mode is "recursive_programming"
    Description: Test AutoParallel(net, parallel_mode = "recursive_programming")
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    init(backend_name='hccl')
    parallel_mode = "recursive_programming"
    context_loss = train_auto_parallel_baseline(parallel_mode)
    context.reset_auto_parallel_context()
    parallel_loss = train_auto_parallel_using_autoparallel_cell(parallel_mode)
    allclose_nparray(np.array(parallel_loss),
                     np.array(context_loss), 0.001, 0.001)
