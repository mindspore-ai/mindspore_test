# Copyright 2021 Huawei Technologies Co., Ltd
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
""" test launch_blocking """
import numpy as np
import pytest
import mindspore.context as context
import mindspore.runtime as rt
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P

context.set_context(mode=context.PYNATIVE_MODE)

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.get_next = P.GetNext([mstype.float32], [(1, 1)], 1, "test")

    def construct(self, x1,):
        x = self.get_next()
        x = x + x1
        return x

def test_launch_blocking_true():
    """
    Feature: test launch blocking
    Description: the operation will be sync
    Expectation: run success
    """
    rt.launch_blocking()
    with pytest.raises(RuntimeError) as execinfo:
        x1 = np.random.randn(1, 1).astype(np.float32)
        net = Net()
        output = net(Tensor(x1))
        print(output.asnumpy())
    assert "GetNext" in str(execinfo.value)

def test_launch_blocking_false():
    """
    Feature: test launch blocking
    Description: the operation will be async
    Expectation: run success
    """
    with pytest.raises(RuntimeError) as execinfo:
        x1 = np.random.randn(1, 1).astype(np.float32)
        net = Net()
        output = net(Tensor(x1))
        print(output.asnumpy())
    assert "Sync stream error" in str(execinfo.value)
