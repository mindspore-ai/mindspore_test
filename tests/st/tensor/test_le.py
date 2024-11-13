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
from mindspore import Tensor

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_le(mode):
    """
    Feature: test Tensor.le.
    Description: Verify the result of Tensor.le..
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x_np = np.array([1, 2, 3]).astype(np.int32)
    y_np = np.array([1, 1, 4]).astype(np.int32)
    x = Tensor(x_np, dtype=ms.int32)
    y = Tensor(y_np, dtype=ms.int32)
    expect_output = x_np <= y_np
    output = x.le(y)
    assert np.allclose(output.asnumpy(), expect_output)

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_less_equal(mode):
    """
    Feature: test Tensor.less_equal.
    Description: Verify the result of Tensor.less_equal.
    Expectation: expect correct forward result.
    """
    ms.set_context(mode=mode)
    x_np = np.array([1, 2, 3]).astype(np.int32)
    y_np = np.array([1, 1, 4]).astype(np.int32)
    x = Tensor(x_np, dtype=ms.int32)
    y = Tensor(y_np, dtype=ms.int32)
    expect_output = x_np <= y_np
    output2 = x.less_equal(y)
    assert np.allclose(output2.asnumpy(), expect_output)
