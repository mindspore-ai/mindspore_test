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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.ops import operations as P
from mindspore import Parameter
from mindspore.common.initializer import initializer
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE,
                    memory_optimize_level='O0')


class TestNet(nn.Cell):
    """Test network"""
    def __init__(self):
        super(TestNet, self).__init__()
        self.fc1_werght = Parameter(initializer('normal', [28*28, 2560*50], ms.float32))
        self.fc2_werght = Parameter(initializer('normal', [2560*50, 2560], ms.float32), device="CPU")
        self.fc3_werght = Parameter(initializer('normal', [2560, 10], ms.float32))

        self.flatten = P.Flatten()
        self.matmul1 = P.MatMul()
        self.relu1 = P.ReLU()
        self.matmul2 = P.MatMul()
        self.relu2 = P.ReLU()
        self.matmul3 = P.MatMul()

    def construct(self, x):
        x = self.flatten(x)
        x = self.matmul1(x, self.fc1_werght)
        x = self.relu1(x)
        x = self.matmul2(x, self.fc2_werght)
        x = self.relu2(x)
        x = self.matmul3(x, self.fc3_werght)
        return x


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_param_offload():
    '''
    Feature: Mem offload
    Description: Test memory offload for specific parameter
    Expectation: Train TestNet success
    '''
    ms.set_seed(1)
    context.set_context(jit_level='O0')
    context.set_context(memory_offload='ON')
    context.set_context(max_device_memory='8GB')

    epoch = 8
    batch_size = 32

    net = TestNet()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)
    loss_fn = nn.CrossEntropyLoss()
    net_with_loss = WithLossCell(net, loss_fn)
    net_with_loss.set_train()
    train_network = TrainOneStepCell(
        net_with_loss, optimizer)  # optimizer
    train_network.set_train()
    losses = []
    for _ in range(0, epoch):
        data = Tensor(np.ones([batch_size, 1, 28, 28]
                              ).astype(np.float32) * 0.01)
        label = Tensor(np.ones([batch_size]).astype(np.int32))
        loss = train_network(data, label)
        losses.append(loss)
    assert losses[-1].asnumpy() <= 2.28684
