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
from mindspore.ops.auto_generate import MoeInitRouting
from parallel.utils.utils import ParallelValidator, compile_net

class MoeInitRoutingNet(Cell):
    def __init__(self, strategy):
        super().__init__()
        self.moeinitrouting = MoeInitRouting().shard(strategy)

    def construct(self, x, rowIdx, expertIdx, activeNum):
        expanded_x, expanded_row_idx, expanded_expert_idx = self.moeinitrouting(x, rowIdx, expertIdx, activeNum)
        return expanded_x, expanded_row_idx, expanded_expert_idx

def test_moe_init_routing_case0():
    """
    Feature: Test moe_init_routing parallel
    Description: shard(1)
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=1, global_rank=0)
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    n = 10
    col = 200
    k = 2
    activeNum = n

    x = Parameter(Tensor(np.random.uniform(-1, 1, size=(n, col)).astype(np.float16)), "x")
    rowIdx = Parameter(Tensor(np.arange(n * k).reshape([k, n]).transpose(1, 0).astype(np.int32)), "rowIdx")
    expertIdx = Parameter(Tensor(np.random.randint(0, 100, size=(n, k)).astype(np.int32)), "expertIdx")
    strategy = ((1, 1), (1, 1), (1, 1))
    moeinitrouting_net = MoeInitRoutingNet(strategy=strategy)
    moeinitrouting_net.set_inputs(x, rowIdx, expertIdx, activeNum)
    phase = compile_net(moeinitrouting_net, x, rowIdx, expertIdx, activeNum)

    validator = ParallelValidator(moeinitrouting_net, phase)
    assert validator.check_parameter_shape('x', [n, col])
    assert validator.check_parameter_shape('rowIdx', [n, k])
    assert validator.check_parameter_shape('expertIdx', [n, k])
