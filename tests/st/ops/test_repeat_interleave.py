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
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor


class RepeatInterleave(nn.Cell):
    def construct(self, x):
        return ops.repeat_interleave(x, repeats=2, axis=0)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_repeat_interleave(mode):
    """
    Feature: tensor.repeat_interleave
    Description: Verify the result of repeat_interleave
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([[0, 1, 2], [3, 4, 5]]), ms.int32)
    net = RepeatInterleave()
    output = net(x)
    expect_output = [[0, 1, 2],
                     [0, 1, 2],
                     [3, 4, 5],
                     [3, 4, 5]]
    assert np.allclose(output.asnumpy(), expect_output)
