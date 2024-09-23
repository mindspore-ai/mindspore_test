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
import pytest
import os
from tests.mark_utils import arg_mark

import mindspore.common.dtype as mstype
import mindspore as ms
import mindspore.nn as nn

class SubNet(nn.Cell):
    def construct(self, x, y, alpha=1):
        return x.sub(y, alpha=alpha)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_f_sub(mode):
    """
    Feature: tensor.sub()
    Description: Verify the result of tensor.sub
    Expectation: success
    """
    os.environ["MS_TENSOR_API_ENABLE_MINT"] = '1'
    ms.set_context(mode=mode)
    net = SubNet()
    x = ms.Tensor([4, 5, 6], dtype=mstype.float32)
    y = ms.Tensor([1, 2, 3], dtype=mstype.float32)
    output = net(x, y)
    expected = np.array([3, 3, 3], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)

    x = ms.Tensor([4, 5, 6], dtype=mstype.float32)
    y = ms.Tensor(3, dtype=mstype.float32)
    alpha = 0.5
    output = net(x, y, alpha=alpha)
    expected = np.array([2.5, 3.5, 4.5], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
    del os.environ["MS_TENSOR_API_ENABLE_MINT"]
