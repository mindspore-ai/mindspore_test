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

import os
import numpy as np

from mindspore import nn, Tensor, context
from mindspore.communication import init, get_rank
from mindspore.nn import Momentum, TrainOneStepCell
from mindspore.ops._grad_experimental.grad_comm_ops import get_squared_device_local_norm_param
import mindspore.ops as P

context.set_context(mode=context.GRAPH_MODE)
init()


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Dense(1, 8)
        self.fc2 = nn.Dense(8, 8)
        self.relu = P.ReLU()

    def construct(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def calc_squared_device_local_norm(rank_dump_path):
    squared_device_local_norm = np.array(0, dtype=np.float32)
    for file in os.listdir(rank_dump_path):
        file = os.path.join(rank_dump_path, file)
        squared_device_local_norm += np.square(np.load(file))
    return squared_device_local_norm


def test_dump_local_norm_and_device_local_norm():
    """
    Feature: Test dump local norm and device local norm
    Description: Test dump local norm and device local norm
    Expectation: local norms match with device local norm on each device
    """
    dump_path = "./dump"
    context.set_auto_parallel_context(device_num=8, dataset_strategy="full_batch", parallel_mode="semi_auto_parallel",
                                      dump_local_norm=True, dump_local_norm_path=dump_path,
                                      dump_device_local_norm=True)
    net = Net()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    net = TrainOneStepCell(net, optimizer)
    net.set_train()
    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    net(x)

    rank_dump_path = dump_path + "/rank_" + str(get_rank())
    squared_norm1 = calc_squared_device_local_norm(rank_dump_path)
    squared_norm2 = get_squared_device_local_norm_param().asnumpy()
    assert np.isclose(squared_norm1, squared_norm2)
