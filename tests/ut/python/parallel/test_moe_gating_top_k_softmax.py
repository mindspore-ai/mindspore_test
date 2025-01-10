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
import pytest
import mindspore as ms
from mindspore import context, Tensor, Parameter
from mindspore.nn import Cell
from mindspore.ops.auto_generate import MoeGatingTopKSoftmax
from parallel.utils.utils import ParallelValidator, compile_net

class MoeGatingTopKSoftmaxNet(Cell):
    def __init__(self, strategy):
        super().__init__()
        self.moegts = MoeGatingTopKSoftmax().shard(strategy)

    def construct(self, x, finished, k):
        out = self.moegts(x, finished, k)
        return out

def test_moe_gating_top_k_softmax_case0():
    """
    Feature: Test moe_gating_top_k_softmax parallel
    Description: shard(1)
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=1, global_rank=0)
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)

    k = 4
    n = 10
    col = 200
    x = Parameter(Tensor(np.random.uniform(-1, 1, size=(n, col)).astype(np.float16)), "x")
    finished = Parameter(Tensor(np.random.uniform(-1, 1, size=(n)).astype(bool)), "finished")
    strategy = ((1, 1), (1,))
    moegts_net = MoeGatingTopKSoftmaxNet(strategy=strategy)
    moegts_net.set_inputs(x, finished, k)
    phase = compile_net(moegts_net, x, finished, k)

    validator = ParallelValidator(moegts_net, phase)
    assert validator.check_parameter_shape('x', [n, col])
    assert validator.check_parameter_shape('finished', [n])

def test_moe_gating_top_k_softmax_error_case():
    """
    Feature: Test moe_gating_top_k_softmax parallel
    Description: Only 1 is supported for the strategy now
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=1, global_rank=0)
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)

    k = 4
    n = 10
    col = 200
    x = Tensor(np.random.uniform(-1, 1, size=(n, col)).astype(np.float16))
    finished = Tensor(np.random.uniform(-1, 1, size=(n)).astype(bool))
    strategy = ((2, 1), (1,))
    moegts_net = MoeGatingTopKSoftmaxNet(strategy=strategy)
    moegts_net.set_inputs(x, finished, k)
    with pytest.raises(RuntimeError):
        compile_net(moegts_net, x, finished, k)
