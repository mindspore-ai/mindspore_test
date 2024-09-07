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
from mindspore.ops.auto_generate import ApplyRotaryPosEmb
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class ApplyRotaryPosEmbNet(Cell):
    def __init__(self, strategy):
        super(ApplyRotaryPosEmbNet, self).__init__()
        self.applyrotaryposemb = ApplyRotaryPosEmb(cos_format=1).shard(strategy)

    def construct(self, query, key, cos, sin, batch_valid_length):
        return self.applyrotaryposemb(query, key, cos, sin, batch_valid_length)

def test_rope():
    """
    Feature: test KVCacheScatterUpdate auto parallel
    Description: auto parallel
    Expectation: shape is as expected.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy = ((2, 1, 4, 1), (2, 1, 4, 1), (1, 1, 1, 1), (1, 1, 1, 1), (1,))

    # numpy input
    query_data = np.random.uniform(0, 1, [4, 1024, 16, 128]).astype(np.float16)

    key_data = np.random.uniform(0, 1, [4, 1024, 16, 128]).astype(np.float16)

    cos_data = np.random.uniform(0, 1, [4, 1024, 1, 128]).astype(np.float16)

    sin_data = np.random.uniform(0, 1, [4, 1024, 1, 128]).astype(np.float16)

    # tensor input
    query = Parameter(Tensor(query_data, ms.float16), 'query')
    key = Parameter(Tensor(key_data, ms.float16), 'key')
    cos = Tensor(cos_data, ms.float16)
    sin = Tensor(sin_data, ms.float16)
    batch_valid_length = Tensor(np.ones((4)), ms.int32)
    net = ApplyRotaryPosEmbNet(strategy)
    net.set_inputs(query, key, cos, sin, batch_valid_length)

    phase = compile_net(net, query, key, cos, sin, batch_valid_length)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('query', [2, 1024, 4, 128])
    assert validator.check_parameter_shape('key', [2, 1024, 4, 128])
