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


class TrilNet(nn.Cell):
    def construct(self, x, diagonal=0):
        return x.tril(diagonal)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_tril(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.tril.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = TrilNet()

    x = ms.Tensor(np.array([[1, 2, 3, 4],
                            [5, 6, 7, 8],
                            [10, 11, 12, 13],
                            [14, 15, 16, 17]]), ms.float32)
    output = net(x)
    expect_output = np.array([[1, 0, 0, 0],
                              [5, 6, 0, 0],
                              [10, 11, 12, 0],
                              [14, 15, 16, 17]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    output = net(x, diagonal=1)
    expect_output = np.array([[1, 2, 0, 0],
                              [5, 6, 7, 0],
                              [10, 11, 12, 13],
                              [14, 15, 16, 17]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
    output = net(x, diagonal=-1)
    expect_output = np.array([[0, 0, 0, 0],
                              [5, 0, 0, 0],
                              [10, 11, 0, 0],
                              [14, 15, 16, 0]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
