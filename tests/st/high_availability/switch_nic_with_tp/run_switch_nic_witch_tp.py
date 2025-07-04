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
import argparse
import  threading

import mindspore.context as context
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import mindspore.nn as nn
from mindspore.common import dtype as mstype
from mindspore.dataset.vision import Inter
from mindspore.train import Model, LossMonitor, Accuracy, Callback
from mindspore.common.initializer import TruncatedNormal
from mindspore.communication import init, get_rank, get_group_size
from mindspore.communication.management import _comm_switch_nic

parser = argparse.ArgumentParser(description='test_ps_lenet')
parser.add_argument("--device_target", type=str, default="Ascend")
parser.add_argument("--dataset_path", type=str, default="/home/workspace/mindspore_dataset/mnist")
args, _ = parser.parse_known_args()
device_target = args.device_target
dataset_path = args.dataset_path
context.set_context(mode=context.GRAPH_MODE, device_target=device_target)


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """Weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")


def fc_with_initialize(input_channels, out_channels):
    """Weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)


def weight_variable():
    """Weight initial"""
    return TruncatedNormal(0.02)


class LeNet5(nn.Cell):
    """
    Define network LeNet5
    """
    def __init__(self, num_class=10, channel=1):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.conv1 = conv(channel, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """
    Create dataset for train or test
    """
    # define dataset
    mnist_ds = ds.MnistDataset(data_path, num_shards=get_group_size(),
                               shard_id=get_rank())

    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Bilinear mode
    rescale_nml_op = CV.Rescale(rescale_nml, shift_nml)
    rescale_op = CV.Rescale(rescale, shift)
    hwc2chw_op = CV.HWC2CHW()
    type_cast_op = C.TypeCast(mstype.int32)

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


class SwitchNicThread(threading.Thread):
    """Create a new thread to switch NIC"""

    def __init__(self, global_ranks, use_backup):
        super().__init__()
        self.func = _comm_switch_nic
        self.global_ranks = global_ranks
        self.use_backup = use_backup

    def run(self):
        print(f"Start switch NIC, Process id: {os.getpid()}, thread id  : {threading.currentThread().ident}")
        self.func(self.global_ranks, self.use_backup)


class CommSwitchNicUseBackup(Callback):
    """
    Comm switch nic use backup
    """

    def __init__(self):
        super(CommSwitchNicUseBackup, self).__init__()
        self.count = 0

    def on_train_step_end(self, run_context):
        """
        Switch nic at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.  For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        print(f"SwitchNicUseBackup callback, Process id: {os.getpid()}, thread id  : {threading.currentThread().ident}")
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        if cur_epoch_num == 1 and self.count == 0:
            t1 = SwitchNicThread([0, 2], [True, True])
            t1.start()
            t1.join()
            _comm_switch_nic([0, 2], [True, True])
            self.count += 1


class CommSwitchNic(Callback):
    """
    Comm switch nic not use backup
    """

    def __init__(self):
        super(CommSwitchNic, self).__init__()
        self.count = 0

    def on_train_step_end(self, run_context):
        """
        Switch nic at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.  For more details,
                    please refer to :class:`mindspore.train.RunContext`.
        """
        print(f"SwitchNic callback, Process id: {os.getpid()}, thread id  : {threading.currentThread().ident}")
        cb_params = run_context.original_args()
        cur_epoch_num = cb_params.get("cur_epoch_num", 1)
        if cur_epoch_num == 2 and self.count == 0:
            t1 = SwitchNicThread([0, 1], [False, True])
            t1.start()
            t1.join()
            self.count += 1


if __name__ == "__main__":
    init()
    context.set_auto_parallel_context(parallel_mode="auto_parallel", gradients_mean=True, device_num=get_group_size())
    network = LeNet5(10)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    net_opt = nn.Momentum(network.trainable_params(), 0.01, 0.9)
    model = Model(network, net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    ds_train = create_dataset(os.path.join(dataset_path, "train"), 32, 1)
    model.train(3, ds_train, callbacks=[LossMonitor(), CommSwitchNicUseBackup(), CommSwitchNic()],
                dataset_sink_mode=True)

    ds_eval = create_dataset(os.path.join(dataset_path, "test"), 32, 1)
    acc = model.eval(ds_eval, dataset_sink_mode=True)

    print("=====Accuracy=====")
    print(acc['Accuracy'])
