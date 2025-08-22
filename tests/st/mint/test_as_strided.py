# Copyright 2025 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.nn as nn
from mindspore import mint
import numpy as np
import pytest

from tests.mark_utils import arg_mark


class AsStridedNet(nn.Cell):
    def construct(self, x, size, strided, storage_offset):
        return mint.as_strided(x, size, strided, storage_offset)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_as_strided_std(mode):
    """
    Feature: mint.as_strided
    Description: Test Tensor feature with mint.as_strided.
    Expectation: Run success
    """
    ms.set_context(mode=mode)
    net = AsStridedNet()
    x = ms.Tensor(np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]), ms.float32)
    output = net(x, (2, 2), (1, 2), 2)
    expect_output = np.array([[3., 5.],
                              [4., 6.]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
