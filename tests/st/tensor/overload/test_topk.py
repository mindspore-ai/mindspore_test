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


class TopkPythonNet(nn.Cell):
    # pylint: disable=redefined-builtin
    def construct(self, x, k, dim=None, largest=True, sorted=True):
        return x.topk(k, dim, largest, sorted)


class TopkPyboostNet(nn.Cell):
    # pylint: disable=redefined-builtin
    def construct(self, x, k, dim=-1, largest=True, sorted=True):
        return x.topk(k, dim, largest, sorted)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_topk_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.topk.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = TopkPythonNet()

    x = ms.Tensor([[0.5368, 0.2447, 0.4302, 0.9673],
                   [0.4388, 0.6525, 0.4685, 0.1868],
                   [0.3563, 0.5152, 0.9675, 0.8230]], ms.float32)
    output = net(x, 2)
    expect_output0 = np.array([[0.9673, 0.5368],
                               [0.6525, 0.4685],
                               [0.9675, 0.823]], dtype=np.float32)
    expect_output1 = np.array([[3, 0],
                               [1, 2],
                               [2, 3]], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect_output0, rtol=1e-3, atol=1e-5)
    assert np.allclose(output[1].asnumpy(), expect_output1, rtol=1e-3, atol=1e-5)
    output = net(x, 2, dim=1)
    expect_output0 = np.array([[0.9673, 0.5368],
                               [0.6525, 0.4685],
                               [0.9675, 0.823]], dtype=np.float32)
    expect_output1 = np.array([[3, 0],
                               [1, 2],
                               [2, 3]], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect_output0, rtol=1e-3, atol=1e-5)
    assert np.allclose(output[1].asnumpy(), expect_output1, rtol=1e-3, atol=1e-5)
    output1 = net(x, 2, dim=1, largest=False)
    expect_output0 = np.array([[2.44700000e-01, 4.30200011e-01],
                               [1.86800003e-01, 4.38800007e-01],
                               [3.56299996e-01, 5.15200019e-01]], dtype=np.float32)
    expect_output1 = np.array([[1, 2],
                               [3, 0],
                               [0, 1]], dtype=np.float32)
    assert np.allclose(output1[0].asnumpy(), expect_output0, rtol=1e-3, atol=1e-5)
    assert np.allclose(output1[1].asnumpy(), expect_output1, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_topk_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.topk.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = TopkPyboostNet()

    x = ms.Tensor([[0.5368, 0.2447, 0.4302, 0.9673],
                   [0.4388, 0.6525, 0.4685, 0.1868],
                   [0.3563, 0.5152, 0.9675, 0.8230]], ms.float32)
    output = net(x, 2)
    expect_output0 = np.array([[0.9673, 0.5368],
                               [0.6525, 0.4685],
                               [0.9675, 0.823]], dtype=np.float32)
    expect_output1 = np.array([[3, 0],
                               [1, 2],
                               [2, 3]], dtype=np.float32)
    assert np.allclose(output[0].asnumpy(), expect_output0, rtol=1e-3, atol=1e-5)
    assert np.allclose(output[1].asnumpy(), expect_output1, rtol=1e-3, atol=1e-5)
    output1 = net(x, 2, dim=1)
    expect_output0 = np.array([[0.9673, 0.5368],
                               [0.6525, 0.4685],
                               [0.9675, 0.823]], dtype=np.float32)
    expect_output1 = np.array([[3, 0],
                               [1, 2],
                               [2, 3]], dtype=np.float32)
    assert np.allclose(output1[0].asnumpy(), expect_output0, rtol=1e-3, atol=1e-5)
    assert np.allclose(output1[1].asnumpy(), expect_output1, rtol=1e-3, atol=1e-5)
    output2 = net(x, 2, dim=1, largest=False)
    expect_output0 = np.array([[2.44700000e-01, 4.30200011e-01],
                               [1.86800003e-01, 4.38800007e-01],
                               [3.56299996e-01, 5.15200019e-01]], dtype=np.float32)
    expect_output1 = np.array([[1, 2],
                               [3, 0],
                               [0, 1]], dtype=np.float32)
    assert np.allclose(output2[0].asnumpy(), expect_output0, rtol=1e-3, atol=1e-5)
    assert np.allclose(output2[1].asnumpy(), expect_output1, rtol=1e-3, atol=1e-5)
