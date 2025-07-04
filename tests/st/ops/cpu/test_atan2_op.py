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
import mindspore.nn as nn
import numpy as np
import pytest

from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetAtan2(nn.Cell):
    def __init__(self):
        super(NetAtan2, self).__init__()
        self.atan2 = P.Atan2()

    def construct(self, x, y):
        return self.atan2(x, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_atan2(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Atan2
    Expectation: the result match to numpy
    """
    np_array = np.array([1, 2, 3, 4, 5]).astype(dtype)
    input_x = Tensor(np_array)
    net = NetAtan2()
    output = net(input_x, input_x)
    print(output)
    expect = np.arctan2(np_array, np_array)
    assert np.allclose(output.asnumpy(), expect)
