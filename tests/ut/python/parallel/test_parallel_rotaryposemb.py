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

import pytest
import numpy as np

from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.ops.auto_generate import RotaryPositionEmbedding
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from parallel.utils.utils import compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


x_ = Tensor(np.random.uniform(-2, 2, (4, 8192, 4, 128)), dtype=mstype.float16)
sin_ = Tensor(np.random.uniform(-1, 1, (4, 8192, 1, 128)), dtype=mstype.float16)
cos_ = Tensor(np.random.uniform(-1, 1, (4, 8192, 1, 128)), dtype=mstype.float16)


class Net(Cell):
    def __init__(self, strategy=None):
        super(Net, self).__init__()
        self.rotarypositionembedding = RotaryPositionEmbedding().shard(strategy)
        self.addn = P.AddN().shard(strategy)

    def construct(self, x, sin, cos):
        q = self.rotarypositionembedding(x, sin, cos)
        return self.rotarypositionembedding(q, sin, cos)


def test_rotarypositionembedding_auto_parallel():
    """
    Feature: test RotaryPosEmb auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_context(save_graphs=True)
    strategy = ((4, 1, 2, 1), (4, 1, 1, 1), (4, 1, 1, 1))
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                      global_rank=0)
    net = Net(strategy)
    compile_net(net, x_, sin_, cos_)

def test_rotarypositionembedding_auto_parallel_strategy_error():
    """
    Feature: test RotaryPosEmb auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    with pytest.raises(RuntimeError) as raise_info:
        strategy = ((4, 1, 1, 2), (4, 1, 1, 2), (4, 1, 1, 2))
        context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                          global_rank=0)
        net = Net(strategy)
        compile_net(net, x_, sin_, cos_)
    assert "RotaryPositionEmbedding init failed" in str(raise_info.value)
