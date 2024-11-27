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
from math import pi
import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x):
        return x.sin()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_sin(mode):
    """
    Feature: tensor.sin
    Description: Verify the result of sin
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([-pi/6, pi/6, pi*10]), ms.float32)
    net = Net()
    output = net(x)
    expect_output = np.array([-0.5, 0.5, 0], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output, rtol=5e-3, atol=1e-4)
