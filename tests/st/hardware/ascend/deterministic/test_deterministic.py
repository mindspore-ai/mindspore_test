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
import os
import pytest
from tests.mark_utils import arg_mark

from mindspore import ops, nn, context
import mindspore as ms
from mindspore.communication import init

class ReduceMatmulNet(nn.Cell):
    def __init__(self, shape):
        super(ReduceMatmulNet, self).__init__()
        self.weight = ms.Parameter(ms.Tensor(np.full(shape, 1), dtype=ms.float32), name="weight")
        self.reducemean = ops.ReduceMean()
        self.matmul = ops.MatMul()
        self.reducesum = ops.ReduceSum()

    def construct(self, x):
        output = self.reducemean(x, 1)
        output = self.matmul(output, self.weight)
        output = self.reducesum(output)
        return output


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'jit'])
def test_deterministic_reduce_matmul(mode):
    """
    Feature: ascend op deterministic test case
    Description: test deterministic for reduce ops and matmul in acl/ge
    Expectation: the result of multiple run should be same
    """
    context.set_context(deterministic="ON")
    x = ms.Tensor(np.random.randn(8090, 4, 8), ms.float32)
    net = ReduceMatmulNet(shape=(8, 4096))
    if mode == 'pynative':
        output1 = net(x)
        output2 = net(x)
    elif mode == 'jit':
        output1 = (jit(net))(x)
        output2 = (jit(net))(x)
    assert np.allclose(output1.asnumpy(), output2.asnumpy(), rtol=0, atol=0)


class AllReduceNet(nn.Cell):
    def __init__(self):
        super(AllReduceNet, self).__init__()
        self.allreduce = ops.AllReduce()

    def construct(self, x):
        output = self.allreduce(x)
        return output


def test_allreduce_deterministic():
    """
    Feature: ascend op deterministic test case
    Description: test deterministic for allreduce
    Expectation: the result of multiple run should be same
    """
    context.set_context(mode=ms.GRAPH_MODE, deterministic="ON")
    init()
    x = ms.Tensor(np.random.randn(16, 1024), ms.float32)
    allreduce_net = AllReduceNet()
    output1 = allreduce_net(x)
    output2 = allreduce_net(x)
    assert np.allclose(output1.asnumpy(), output2.asnumpy(), rtol=0, atol=0)


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="allcards", essential_mark="essential")
def test_deterministic_allreduce():
    """
    Feature: mpirun ascend op deterministic test case
    Description: test deterministic for allreduce
    Expectation: the result of multiple run should be same
    """
    return_code = os.system("mpirun --allow-run-as-root -n 8 pytest -s test_deterministic.py::" \
                            "test_allreduce_deterministic")
    assert return_code == 0
