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
from mindspore import nn, ops
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

class DenseL2(nn.Cell):
    def __init__(self, in_channels, hidden_size):
        super(DenseL2, self).__init__()
        self.dense1 = nn.Dense(in_channels, hidden_size, weight_init="ones", has_bias=False)
        self.bias = Parameter(initializer("zeros", [hidden_size], self.dense1.weight.dtype))
        self.add = ops.BiasAdd()

    def construct(self, x):
        x = self.dense1(x)
        x = self.add(x, self.bias)
        return x

class DenseL3(nn.Cell):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(DenseL3, self).__init__()
        self.block = DenseL2(in_channels, hidden_size)
        self.dense2 = nn.Dense(hidden_size, out_channels, weight_init="ones", has_bias=False)

    def construct(self, x):
        x = self.block(x)
        x = self.dense2(x)
        return x

class SlimLeNet(nn.Cell):
    def __init__(self):
        super(SlimLeNet, self).__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.SequentialCell(
            nn.Dense(28*28, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 512, weight_init="normal", bias_init="zeros"),
            nn.ReLU(),
            nn.Dense(512, 10, weight_init="normal", bias_init="zeros")
        )

    def construct(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

class DenseNet(nn.Cell):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(DenseNet, self).__init__()
        self.dense1 = nn.Dense(in_channels, hidden_size, weight_init="ones", has_bias=False)
        self.dense2 = nn.Dense(hidden_size, out_channels, weight_init="ones", has_bias=False)

    def construct(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.reduce_partial()
        return x
