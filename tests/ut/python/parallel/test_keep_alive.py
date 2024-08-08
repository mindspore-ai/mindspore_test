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
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops import operations as P
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

class GatherNet(Cell):
    def __init__(self, weight, strategy1=None, strategy2=None, strategy3=None):
        super().__init__()
        self.sub = P.Sub().shard(strategy1)
        self.weight = Parameter(weight, "w1")
        self.sqrt = P.Sqrt().shard(strategy2)
        self.relu = P.ReLU().shard(strategy2)

        self.add = P.Add().shard(strategy3)
        self.a = Tensor(np.array([0, 1, 2, 3, 4, 5, 6, 7]), dtype=ms.float32)

    def construct(self, x):
        x = self.relu(x)
        out = self.sub(x, self.weight)
        bias = self.a
        out = self.sqrt(out)
        out = self.add(out, bias)
        return out


def test_keep_alive():
    """
    Feature: test keep alive
    Description:
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    context.set_context(save_graphs=True)
    strategy1 = ((8, 1, 1), (8, 1, 1))
    strategy2 = ((1, 1, 8),)
    strategy3 = ((1, 1, 8), (8,))
    weight = Tensor(np.ones([8, 32, 8]), dtype=ms.float32)
    net = GatherNet(weight, strategy1, strategy2, strategy3)
    input_x = Tensor(np.ones([8, 32, 8]), dtype=ms.float32)

    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs_has('Add-0', ['Sqrt-0'])
