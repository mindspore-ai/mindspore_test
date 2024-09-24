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
from tests.mark_utils import arg_mark

import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, dim, keepdim):
        return x.mean(dim, keepdim)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_mean(mode):
    """
    Feature: tensor.mean
    Description: Verify the result of tensor.mean
    Expectation: success
    """
    os.environ["MS_TENSOR_API_ENABLE_MINT"] = '1'
    ms.set_context(mode=mode)
    x = Tensor(np.array([[0.0, 0.2, 0.4, 0.5, 0.1],
                         [3.2, 0.4, 0.1, 2.9, 4.0]]), ms.float32)
    net = Net()
    output = net(x, 0, True)
    expect_output = Tensor([1.6, 0.3, 0.25, 1.7, 2.05])
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())

    del os.environ["MS_TENSOR_API_ENABLE_MINT"]
