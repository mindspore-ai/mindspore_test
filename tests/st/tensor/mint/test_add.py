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

import os
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from tests.mark_utils import arg_mark

class AddNet(nn.Cell):
    def construct(self, x, other, *, alpha=2):
        out = x.add(other, alpha=alpha)
        return out

class AddNetWoAlpha(nn.Cell):
    def construct(self, x, other):
        out = x.add(other)
        return out

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE])
def test_add(mode):
    """
    Feature: Tensor.add.
    Description: Verify the result of add.
    Expectation: expect correct result.
    """
    net = AddNet()
    net_wo_a = AddNetWoAlpha()
    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    b = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    y = ms.Tensor(b, dtype=ms.float32)
    ms.set_context(mode=mode)
    os.environ["MS_TENSOR_API_ENABLE_MINT"] = '1'
    res = net(x, y, alpha=2.1)
    res1 = net(x, y)
    res2 = net_wo_a(x, y)
    expect = a + 2.1 * b
    expect1 = a + 2 * b
    expect2 = a + b
    np.testing.assert_allclose(res.asnumpy(), expect, rtol=1e-4)
    np.testing.assert_allclose(res1.asnumpy(), expect1, rtol=1e-4)
    np.testing.assert_allclose(res2.asnumpy(), expect2, rtol=1e-4)
    del os.environ["MS_TENSOR_API_ENABLE_MINT"]
