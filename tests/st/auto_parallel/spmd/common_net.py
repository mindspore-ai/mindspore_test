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
"""common net"""
from mindspore import nn, mint
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer

class DenseL2(nn.Cell):
    """dense cell with level 2 depth"""
    def __init__(self, in_channels, hidden_size):
        super().__init__()
        self.dense1 = nn.Dense(in_channels, hidden_size, weight_init="normal", has_bias=False)
        self.bias = Parameter(initializer("zeros", [hidden_size], self.dense1.weight.dtype))

    def construct(self, x):
        x = self.dense1(x)
        x = mint.add(x, self.bias)
        return x

class DenseL3(nn.Cell):
    """dense cell with level 3 depth"""
    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()
        self.block = DenseL2(in_channels, hidden_size)
        self.dense2 = nn.Dense(hidden_size, out_channels, weight_init="normal", has_bias=False)

    def construct(self, x):
        x = self.block(x)
        x = self.dense2(x)
        return x

class SlimLeNet(nn.Cell):
    """slim lenet"""
    def __init__(self):
        super().__init__()
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
    """dense net with 2 dense cell"""
    def __init__(self, in_channels, out_channels, hidden_size):
        super().__init__()
        self.dense1 = nn.Dense(in_channels, hidden_size, weight_init="normal", has_bias=False)
        self.dense2 = nn.Dense(hidden_size, out_channels, weight_init="normal", has_bias=False)

    def construct(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = x.reduce_partial()
        return x

class DenseMutiLayerNet(nn.Cell):
    """dense net with configurable layer number"""
    def __init__(self, hidden_size, layer_num):
        super().__init__()
        self.layer_num = layer_num
        self.layers = nn.CellList()
        for _ in range(self.layer_num):
            layer = DenseL2(hidden_size, hidden_size)
            self.layers.append(layer)

    def construct(self, x):
        for i in range(self.layer_num):
            x = self.layers[i](x)
        return x
