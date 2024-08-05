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

import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def construct(self, x, y):
        output = x.bitwise_and(y)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_x86_gpu_training
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bitwise_and(mode):
    """
    Feature: Tensor.bitwise_and
    Description: Verify the result of bitwise_and
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = ms.Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int32))
    y = ms.Tensor(np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]], dtype=np.int32))
    expect_output = np.array([[1, 0, 3], [4, 5, 4], [3, 0, 1]], dtype=np.int32)
    out = net(x, y)
    assert np.allclose(out.asnumpy(), expect_output)
    out2 = x & y
    assert np.allclose(out2.asnumpy(), expect_output)
