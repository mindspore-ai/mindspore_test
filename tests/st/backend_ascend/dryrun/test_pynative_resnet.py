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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import Dense
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.cell import Cell
from mindspore.nn.layer.basic import Flatten
from mindspore.nn.layer.conv import Conv2d
from mindspore.nn.layer.normalization import BatchNorm2d
from mindspore.nn.layer.pooling import MaxPool2d
from mindspore.nn.optim import Momentum
from mindspore.ops import operations as P
from mindspore.ops.operations import Add
from mindspore.communication.management import init
from mindspore.parallel import set_algo_parameters

context.set_context(mode=context.PYNATIVE_MODE)


def random_normal_init(shape, mean=0.0, stddev=0.01, seed=None):
    """random_normal_init"""
    init_value = np.ones(shape).astype(np.float32) * 0.01
    return Tensor(init_value)


def variance_scaling_raw(shape):
    """variance_scaling_raw"""
    variance_scaling_value = np.ones(shape).astype(np.float32) * 0.01
    return Tensor(variance_scaling_value)


def weight_variable_0(shape):
    """weight_variable_0"""
    zeros = np.zeros(shape).astype(np.float32)
    return Tensor(zeros)


def weight_variable_1(shape):
    """weight_variable_1"""
    ones = np.ones(shape).astype(np.float32)
    return Tensor(ones)


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    """3x3 convolution """
    weight_shape = (out_channels, in_channels, 3, 3)
    weight = variance_scaling_raw(weight_shape)
    return Conv2d(in_channels, out_channels,
                  kernel_size=3, stride=stride, weight_init=weight, has_bias=False, pad_mode="same")


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    weight_shape = (out_channels, in_channels, 1, 1)
    weight = variance_scaling_raw(weight_shape)
    return Conv2d(in_channels, out_channels,
                  kernel_size=1, stride=stride, weight_init=weight, has_bias=False, pad_mode="same")


def conv7x7(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    weight_shape = (out_channels, in_channels, 7, 7)
    weight = variance_scaling_raw(weight_shape)
    return Conv2d(in_channels, out_channels,
                  kernel_size=7, stride=stride, weight_init=weight, has_bias=False, pad_mode="same")


def bn_with_initialize(out_channels):
    """bn_with_initialize"""
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    gamma = weight_variable_1(shape)
    bn = BatchNorm2d(out_channels, momentum=0.1, eps=0.0001, gamma_init=gamma,
                     beta_init=beta, moving_mean_init=mean, moving_var_init=var)
    return bn


def bn_with_initialize_last(out_channels):
    """bn_with_initialize_last"""
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    gamma = weight_variable_0(shape)
    bn = BatchNorm2d(out_channels, momentum=0.1, eps=0.0001, gamma_init=gamma,
                     beta_init=beta, moving_mean_init=mean, moving_var_init=var)
    return bn


def fc_with_initialize(input_channels, out_channels):
    """fc_with_initialize"""
    weight_shape = (out_channels, input_channels)
    bias_shape = (out_channels)
    weight = random_normal_init(weight_shape)
    bias = weight_variable_0(bias_shape)

    return Dense(input_channels, out_channels, weight, bias)


class ResidualBlock(Cell):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=1, padding=0)
        self.bn1 = bn_with_initialize(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=stride, padding=1)
        self.bn2 = bn_with_initialize(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = bn_with_initialize_last(out_channels)

        self.relu = P.ReLU()
        self.add = Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResidualBlockWithDown(Cell):
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        super(ResidualBlockWithDown, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=1, padding=0)
        self.bn1 = bn_with_initialize(out_chls)

        self.conv2 = conv3x3(out_chls, out_chls, stride=stride, padding=1)
        self.bn2 = bn_with_initialize(out_chls)

        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = bn_with_initialize_last(out_channels)

        self.relu = P.ReLU()
        self.downSample = down_sample

        self.conv_down_sample = conv1x1(
            in_channels, out_channels, stride=stride, padding=0)
        self.bn_down_sample = bn_with_initialize(out_channels)
        self.add = Add()

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.conv_down_sample(identity)
        identity = self.bn_down_sample(identity)

        out = self.add(out, identity)
        out = self.relu(out)

        return out


class MakeLayer0(Cell):

    def __init__(self, block, layer_num, in_channels, out_channels, stride):
        super(MakeLayer0, self).__init__()
        self.a = ResidualBlockWithDown(
            in_channels, out_channels, stride=1, down_sample=True)
        self.c = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        x = self.a(x)
        x = self.c(x)

        return x

class ResNet(Cell):

    def __init__(self, block, layer_num, num_classes=100):
        super(ResNet, self).__init__()

        self.conv1 = conv7x7(3, 64, stride=2, padding=3)

        self.bn1 = bn_with_initialize(64)
        self.relu = P.ReLU()
        self.maxpool = MaxPool2d(kernel_size=3, stride=2, pad_mode="same")

        self.layer1 = MakeLayer0(
            block, layer_num[0], in_channels=64, out_channels=2048, stride=1)

        self.pool = nn.AvgPool2d(7, 1)
        self.fc = fc_with_initialize(512 * block.expansion, num_classes)
        self.flatten = Flatten()

    def construct(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def resnet50(num_classes):
    return ResNet(ResidualBlock, [1, 1, 1, 1], num_classes)


def train(num_classes=10, epoch=3, batch_size=1):
    import os
    device_id = int(os.getenv('DEVICE_ID', '0'))
    mindspore.set_context(device_id=device_id)
    mindspore.set_auto_parallel_context(device_num=8,
                                        gradients_mean=True,
                                        parallel_mode=mindspore.ParallelMode.DATA_PARALLEL)
    set_algo_parameters(elementwise_op_strategy_follow=True)
    init()
    net = resnet50(num_classes)
    lr = 0.1
    momentum = 0.9
    optimizer = Momentum(filter(lambda x: x.requires_grad,
                                net.get_parameters()), lr, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='sum')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(
        net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    losses = []
    for _ in range(0, epoch):
        data = Tensor(np.ones([batch_size, 3, 224, 224]
                              ).astype(np.float32) * 0.01)
        label = Tensor(np.ones([batch_size]).astype(np.int32))
        loss = train_network(data, label)
        losses.append(loss)


train()
