# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn

class Net(nn.Cell):
    def construct(self, tensor):
        return tensor.stride()

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_stride(mode):
    """
    Feature: tensor.stride
    Description: the stride of tensor
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    net = Net()
    output = net(x)
    expect_out = [5, 1]
    assert np.allclose(output, expect_out)
