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


class SumPythonNet(nn.Cell):
    def construct(self, x, axis=None, dtype=None, keepdims=False, initial=None):
        return x.sum(axis, dtype, keepdims, initial)


class SumPyboostNet(nn.Cell):
    def construct(self, x, dim=None, keepdim=False, dtype=None):
        return x.sum(dim, keepdim, dtype)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_sum_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.sum.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = SumPythonNet()

    x = ms.Tensor([[[1, 2, 3], [2, 3, 4]]], ms.float32)
    output = net(x)
    expect_output = 15.0
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, axis=[0, 1], keepdims=True)
    expect_output = np.array([[[3., 5., 7.]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, axis=(2,), keepdims=False)
    expect_output = np.array([[6., 9.]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, axis=2, keepdims=True)
    expect_output = np.array([[[6.], [9.]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, dtype=ms.bool_, initial=12)
    expect_output = True
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_sum_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.sum.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = SumPyboostNet()

    x = ms.Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
                            [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                            [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), ms.float32)
    output = net(x)
    expect_output = 270.0
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, dim=2)
    expect_output = np.array([[6., 12., 18.],
                              [24., 30., 36.],
                              [42., 48., 54.]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    output = net(x, dim=2, keepdim=True)
    expect_output = np.array([[[6.],
                               [12.],
                               [18.]],
                              [[24.],
                               [30.],
                               [36.]],
                              [[42.],
                               [48.],
                               [54.]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)
