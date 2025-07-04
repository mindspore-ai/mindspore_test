# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor
from mindspore.common.api import _pynative_executor


class Net(nn.Cell):
    def construct(self, x, batch1, batch2, alpha=0.1, beta=0.5):
        output = ops.addbmm(x, batch1, batch2, alpha=alpha, beta=beta)
        return output


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_real_normal():
    """
    Feature: ops.addbmm
    Description: Test 4D input
    Expectation: raise ValueError
    """
    x = Tensor(np.random.randn(6, 8).astype(np.float32))
    b1 = Tensor(np.random.randn(12, 10, 6, 4).astype(np.float32))
    b2 = Tensor(np.random.randn(12, 8, 4, 8).astype(np.float32))
    net = Net()
    with pytest.raises(ValueError):
        net(x, b1, b2)
        _pynative_executor.sync()
