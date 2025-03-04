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

import os
import math
import numpy as np
import mindspore as ms
from mindspore.communication.management import init
from mindspore import nn, ops, context, Tensor
from mindspore.train import LossMonitor, Model
from mindspore.context import ParallelMode
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.communication.management import get_rank
from mindspore.communication.management import get_group_size
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer, HeUniform


class FakeDataInitMode:
    RandomInit = 0
    OnesInit = 1
    UniqueInit = 2
    ZerosInit = 3


class FakeData:
    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224),
                 num_classes=10, random_offset=0, use_parallel=False,
                 fakedata_mode=FakeDataInitMode.RandomInit):
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_classes = num_classes
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = np.float32
        self.label_data_type = np.float32
        self.is_onehot = True
        self.fakedata_mode = fakedata_mode

        if use_parallel:
            init(backend_name='hccl')
            self.rank_size = get_group_size()
            self.rank_id = get_rank()

        self.total_batch_size = self.rank_batch_size * self.rank_size

        assert (self.size % self.total_batch_size) == 0

        self.total_batch_data_size = (
            self.rank_size, self.rank_batch_size) + image_size

    def get_dataset_size(self):
        return int(self.size / self.total_batch_size)

    def get_repeat_count(self):
        return 1

    def set_image_data_type(self, data_type):
        self.image_data_type = data_type

    def set_label_data_type(self, data_type):
        self.label_data_type = data_type

    def set_label_onehot(self, is_onehot=True):
        self.is_onehot = is_onehot

    def create_tuple_iterator(self, num_epochs=-1, do_copy=True):
        _ = num_epochs
        return self

    def __getitem__(self, batch_index):
        if batch_index * self.total_batch_size >= len(self):
            raise IndexError("{} index out of range".format(
                self.__class__.__name__))
        rng_state = np.random.get_state()
        np.random.seed(batch_index + self.random_offset)
        if self.fakedata_mode == FakeDataInitMode.OnesInit:
            img = np.ones(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.ZerosInit:
            img = np.zeros(self.total_batch_data_size)
        elif self.fakedata_mode == FakeDataInitMode.UniqueInit:
            total_size = 1
            for i in self.total_batch_data_size:
                total_size = total_size * i
            img = np.reshape(np.arange(total_size) * 0.001,
                             self.total_batch_data_size)
        else:
            img = np.random.randn(*self.total_batch_data_size)

        np.random.set_state(rng_state)
        img = img[self.rank_id]
        img_ret = img.astype(self.image_data_type)

        total_size = self.rank_batch_size * self.num_classes
        target = np.reshape(np.arange(total_size)*0.001,
                            (self.rank_batch_size, self.num_classes))
        return Tensor(img_ret), Tensor(target, dtype=ms.float32)

    def __len__(self):
        return self.size

    def __iter__(self):
        self.batch_index = 0
        return self

    def reset(self):
        self.batch_index = 0

    def __next__(self):
        if self.batch_index * self.total_batch_size < len(self):
            data = self[self.batch_index]
            self.batch_index += 1
            return data
        raise StopIteration


class ParallelNetwork(nn.Cell):
    """ParallelNetwork"""

    def __init__(self, strategy=None):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = ms.Parameter(initializer(HeUniform(math.sqrt(5)), shape=[
            16, 10], dtype=ms.float32), name="fc1")
        self.matmul1 = ops.MatMul().shard(strategy)
        self.relu1 = ops.ReLU()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_weight)
        x = self.relu1(x)
        return x


class CustomLossMonitor(LossMonitor):
    def __init__(self, per_print_times=1):
        super(CustomLossMonitor, self).__init__(
            per_print_times=per_print_times)
        self.loss_history = []

    def step_end(self, run_context):
        super(CustomLossMonitor, self).step_end(run_context)
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        self.loss_history.append(loss)


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
        format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol,
                           atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert True


def clean_all_ckpt_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                os.remove(os.path.join(folder_path, file_name))


def train_functional_programming_using_autoparallel_cell(strategy):
    ms.set_seed(1)
    dataset = FakeData(size=32, batch_size=8, image_size=(
        4, 4), num_classes=10, fakedata_mode=FakeDataInitMode.UniqueInit)
    ir_path = "model_parallel_ir/parallel_functional_programming_ir_" + \
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

    def train_one_step(inputs, targets):
        loss_value, grads = grad_fn(inputs, targets)
        optimizer(grads)
        return loss_value

    parallel_net = AutoParallel(train_one_step, parallel_mode="semi_auto")
    print("the loss when training using AutoParallel(cell) is :")
    loss_history = []
    for epoch in range(1, 3):
        step = 1
        for input_x, label in dataset:
            loss = parallel_net(input_x, label)
            print("epoch: " + str(epoch) + " step: " +
                  str(step) + " loss: " + str(loss))
            loss_history.append(loss)
            step += 1
    return loss_history


def train_model_programming_using_autoparallel_cell(strategy):
    ms.set_seed(1)
    parallel_ir_path = "model_parallel_ir/parallel_Model_ir_" + str(strategy)
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path=parallel_ir_path)
    parallel_dataset = FakeData(size=32, batch_size=8, image_size=(
        4, 4), num_classes=10, fakedata_mode=FakeDataInitMode.UniqueInit)
    with no_init_parameters():
        net = ParallelNetwork(strategy=strategy)
        optimizer = nn.Momentum(net.trainable_params(),
                                learning_rate=0.1, momentum=0.9)
    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
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
    ir_path = "model_parallel_ir/context_Model_ir_" + str(strategy)
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend',
                        save_graphs=True, save_graphs_path=ir_path)
    context.set_auto_parallel_context(
        parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL)
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


def test_data_parallel_model_programming():
    """
    Feature: AutoParallel(cell) in data parallel dimension
    Description: Train in Model.train way using AutoParallel(cell)
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    init(backend_name='hccl')
    strategy = ((8, 1), (1, 1))
    context_loss = train_model_programming_baseline(strategy)
    context.reset_auto_parallel_context()
    parallel_loss = train_model_programming_using_autoparallel_cell(strategy)
    allclose_nparray(np.array(parallel_loss),
                     np.array(context_loss), 0.001, 0.001)


def test_data_parallel_functional_programming():
    """
    Feature: AutoParallel(cell) in data parallel dimension
    Description: Train in functional programming way using AutoParallel(cell)
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    init(backend_name='hccl')
    strategy = ((8, 1), (1, 1))
    context_loss = train_model_programming_baseline(strategy)
    context.reset_auto_parallel_context()
    parallel_loss = train_functional_programming_using_autoparallel_cell(
        strategy)
    allclose_nparray(np.array(parallel_loss),
                     np.array(context_loss), 0.001, 0.001)


def test_model_parallel_model_programming():
    """
    Feature: AutoParallel(cell) in model parallel dimension
    Description: Train in Model.train way using AutoParallel(cell)
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    init(backend_name='hccl')
    strategy = ((1, 1), (1, 2))
    context_loss = train_model_programming_baseline(strategy)
    context.reset_auto_parallel_context()
    parallel_loss = train_model_programming_using_autoparallel_cell(strategy)
    allclose_nparray(np.array(parallel_loss),
                     np.array(context_loss), 0.001, 0.001)


def test_model_parallel_functional_programming():
    """
    Feature: AutoParallel(cell) in model parallel dimension
    Description: Train in functional programming way using AutoParallel(cell)
    Expectation: The difference between the new loss and the baseline loss is in line with expectations.
    """
    init(backend_name='hccl')
    strategy = ((1, 1), (1, 2))
    context_loss = train_model_programming_baseline(strategy)
    context.reset_auto_parallel_context()
    parallel_loss = train_functional_programming_using_autoparallel_cell(
        strategy)
    allclose_nparray(np.array(parallel_loss),
                     np.array(context_loss), 0.001, 0.001)
