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
import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, dims):
        return x.permute(dims)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_permute(mode):
    """
    Feature: tensor.permute
    Description: Verify the result of permute
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4), ms.float32)
    input_perm = (0, 2, 1)
    net = Net()
    output = net(x, input_perm)
    expect_output = [[[0, 4, 8],
                      [1, 5, 9],
                      [2, 6, 10],
                      [3, 7, 11]],
                     [[12, 16, 20],
                      [13, 17, 21],
                      [14, 18, 22],
                      [15, 19, 23]]]
    assert np.allclose(output.asnumpy(), expect_output)
