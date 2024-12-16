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

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_dist_normal(mode):
    """
    Feature: dist
    Description: Verify the result of norm
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor([[1, 2, 3], [4, 5, 6]], dtype=ms.float32)
    y = ms.Tensor([[6, 5, 4], [3, 2, 1]], dtype=ms.float32)
    p = 2
    output = x.dist(y, p)
    expect_output = np.array([8.3666], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    p = 1
    output = x.dist(y, p)
    expect_output = np.array([18.0], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
