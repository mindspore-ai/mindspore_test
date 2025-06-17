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

import mindspore.nn as nn
from mindspore.common import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P


class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 32
        weight1 = Tensor(np.ones([6, 3, 5, 5]).astype(np.float32) * 0.01)
        weight2 = Tensor(np.ones([16, 6, 5, 5]).astype(np.float32) * 0.01)
        self.conv1 = nn.Conv2d(3, 6, (5, 5), weight_init=weight1, stride=1, padding=0, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, (5, 5), weight_init=weight2, pad_mode='valid', stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="valid")
        self.flatten = nn.Flatten()

        weight1 = Tensor(np.ones([120, 400]).astype(np.float32) * 0.01)
        self.fc1 = nn.Dense(400, 120, weight_init=weight1)

        weight2 = Tensor(np.ones([84, 120]).astype(np.float32) * 0.01)
        self.fc2 = nn.Dense(120, 84, weight_init=weight2)

        weight3 = Tensor(np.ones([10, 84]).astype(np.float32) * 0.01)
        self.fc3 = nn.Dense(84, 10, weight_init=weight3)

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.flatten(output)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output


def multi_step_lr(total_steps, gap, base_lr=0.9, gamma=0.1, dtype=mstype.float32):
    lr = []
    for step in range(total_steps):
        lr_ = base_lr * gamma ** (step // gap)
        lr.append(lr_)
    return Tensor(np.array(lr), dtype)
