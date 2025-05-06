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
from mindspore.communication.management import init
from mindspore import nn, context
from mindspore.train import Model
from mindspore.context import ParallelMode
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.nn.utils import no_init_parameters
from .model_parallel import FakeDataInitMode, FakeData, ParallelNetwork, CustomLossMonitor, allclose_nparray


def train_functional_programming_using_autoparallel_cell(strategy):
    ms.set_seed(1)
    dataset = FakeData(size=32, batch_size=8, image_size=(
        4, 4), num_classes=10, fakedata_mode=FakeDataInitMode.UniqueInit)
    ir_path = "optimizer_parallel_ir/parallel_functional_programming_ir_" + \
        str(strategy)
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path=ir_path)

    with no_init_parameters():
        net = ParallelNetwork(strategy=strategy)
        optimizer = nn.Momentum(net.trainable_params(),
                                learning_rate=0.1, momentum=0.9)
    loss_fn = nn.MSELoss(reduction='mean')

    def forward_fn(data, target):
        logits = net(data)
        loss = loss_fn(logits, target)
        return loss
    grad_fn = ms.value_and_grad(forward_fn, None, net.trainable_params())

    @ms.jit
    def train_one_step(inputs, targets):
        loss_value, grads = grad_fn(inputs, targets)
        optimizer(grads)
        return loss_value

    parallel_net = AutoParallel(train_one_step, parallel_mode="semi_auto")
    parallel_net.hsdp(threshold=0, optimizer_level="level3")
    print("the loss when training using AutoParallel(cell) is :")
    loss_history = []
    for epoch in range(1, 3):
        step = 1
        for input_x, label in dataset:
            parallel_net.compile(input_x, label)
            loss = parallel_net(input_x, label)
            print("epoch: " + str(epoch) + " step: " +
                  str(step) + " loss: " + str(loss))
            loss_history.append(loss)
            step += 1
    return loss_history


def train_model_programming_using_autoparallel_cell(strategy):
    ms.set_seed(1)
    parallel_ir_path = "optimizer_parallel_ir/parallel_Model_ir_" + str(strategy)
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path=parallel_ir_path)
    with no_init_parameters():
        net = ParallelNetwork(strategy=strategy)
        optimizer = nn.Momentum(net.trainable_params(),
                                learning_rate=0.1, momentum=0.9)
    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
    parallel_net.hsdp(threshold=0, optimizer_level="level3")
    parallel_dataset = FakeData(size=32, batch_size=8, image_size=(
        4, 4), num_classes=10, fakedata_mode=FakeDataInitMode.UniqueInit)
    loss_fn = nn.MSELoss(reduction='mean')
    loss_monitor = CustomLossMonitor(per_print_times=1)
    model = Model(network=parallel_net, loss_fn=loss_fn, optimizer=optimizer)
    print("the loss when training using AutoParallel(cell) is :")
    model.train(epoch=2, train_dataset=parallel_dataset,
                dataset_sink_mode=False, callbacks=[loss_monitor])
    return loss_monitor.loss_history


def train_model_programming_baseline(strategy):
    ms.set_seed(1)
    dataset = FakeData(size=32, batch_size=8, image_size=(
        4, 4), num_classes=10, fakedata_mode=FakeDataInitMode.UniqueInit)
    ir_path = "optimizer_parallel_ir/context_Model_ir_" + str(strategy)
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path=ir_path)
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
        enable_parallel_optimizer=True,
        parallel_optimizer_config={"parallel_optimizer_threshold": 0,
                                   "optimizer_level": "level3"})
    net = ParallelNetwork(strategy=strategy)
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = nn.Momentum(net.trainable_params(),
                            learning_rate=0.1, momentum=0.9)
    loss_monitor = CustomLossMonitor(per_print_times=1)
    model = Model(network=net, loss_fn=loss_fn, optimizer=optimizer)
    print("the baseline loss is :")
    model.train(epoch=2, train_dataset=dataset,
                dataset_sink_mode=False, callbacks=[loss_monitor])
    return loss_monitor.loss_history


def test_optimizer_parallel_model_programming():
    """
    Feature: AutoParallel(cell) in data parallel dimension
    Description: Train in Model.train way using AutoParallel(cell)
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    init(backend_name='hccl')
    strategy = ((8, 1), (1, 1))
    parallel_loss = train_model_programming_using_autoparallel_cell(strategy)
    context.reset_auto_parallel_context()
    context_loss = train_model_programming_baseline(strategy)
    allclose_nparray(np.array(parallel_loss),
                     np.array(context_loss), 0.001, 0.001)


def test_optimizer_parallel_functional_programming():
    """
    Feature: AutoParallel(cell) in data parallel dimension
    Description: Train in functional programming way using AutoParallel(cell)
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    init(backend_name='hccl')
    strategy = ((8, 1), (1, 1))
    parallel_loss = train_functional_programming_using_autoparallel_cell(
        strategy)
    context.reset_auto_parallel_context()
    context_loss = train_model_programming_baseline(strategy)
    allclose_nparray(np.array(parallel_loss),
                     np.array(context_loss), 0.001, 0.001)
