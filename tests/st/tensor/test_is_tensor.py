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
from tests.mark_utils import arg_mark

import pytest

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x):
        return ms.is_tensor(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_is_tensor(mode):
    """
    Feature: mindspore.is_tensor
    Description: Verify the result of mindspore.is_tensor
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    a = Tensor([1, 2, 3])
    output1 = net(a)
    assert output1
    b = [1, 2, 3]
    output2 = net(b)
    assert not output2
