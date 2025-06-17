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
import mindspore.nn as nn
from mindspore import context
from mindspore.common import Tensor
from mindspore.nn.optim import Momentum
from lenet_train import LeNet, multi_step_lr


def train_lenet():
    """Train lenet by cpu data parallel"""
    context.set_context(mode=context.GRAPH_MODE)
    ms.set_device(device_target="CPU")
    epoch = 10

    net = LeNet()
    learning_rate = multi_step_lr(epoch, 2)
    momentum = 0.9
    mom_optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = nn.WithLossCell(net, criterion)
    train_network = nn.TrainOneStepCell(net_with_criterion, mom_optimizer)
    train_network.set_train()

    data = Tensor(np.ones([net.batch_size, 3, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([net.batch_size]).astype(np.int32))
    train_network(data, label)
    if not hasattr(LeNet.construct, "source"):
        raise ValueError("Set source code for LeNet failed!")
    print(f"=======LeNet construct source code: {getattr(LeNet.construct, 'source')}=======")

train_lenet()
