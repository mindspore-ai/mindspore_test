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
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):

    def construct(self, start, end, weight):
        output = start.lerp(end, weight)
        return output


@arg_mark(plat_marks=[
    'cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
    'platform_ascend910b'
],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_lerp_pyboost(mode):
    """
    Feature: tensor.lerp
    Description: Verify the result of lerp in pyboost
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    start = Tensor(np.array([1., 2., 3., 4.]), ms.float32)
    end = Tensor(np.array([10., 10., 10., 10.]), ms.float32)
    weight = 0.5
    net = Net()
    expect_out = np.array([5.5, 6., 6.5, 7.])
    assert np.allclose(np.asarray(net(start, end, weight)), expect_out)

    weight = Tensor(np.array([0.5, 0.5, 0.5, 0.5]), ms.float32)
    assert np.allclose(np.asarray(net(start, end, weight)), expect_out)
