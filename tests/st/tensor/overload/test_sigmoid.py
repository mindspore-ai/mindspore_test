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
"""Test the overload functional method"""
import mindspore as ms
import mindspore.nn as nn
import numpy as np
import pytest

from tests.mark_utils import arg_mark


class SigmoidNet(nn.Cell):
    def construct(self, x):
        return x.sigmoid()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_sigmoid(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.sigmoid.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = SigmoidNet()
    x = ms.Tensor(np.array([1, 2, 3, 4, 5]), ms.float32)
    output = net(x)
    expect_output = np.array([0.7310586, 0.880797, 0.95257413, 0.98201376, 0.9933072], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
