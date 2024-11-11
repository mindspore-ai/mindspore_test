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


class MinPythonNet(nn.Cell):
    def construct(self, x, axis=None, keepdims=False, initial=None, where=None, return_indices=False):
        return x.min(axis, keepdims, initial=initial, where=where, return_indices=return_indices)


class MinPyboostNet(nn.Cell):
    def construct(self, x):
        return x.min()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_min_python(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.min.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    # test 1: using positional args
    net = MinPythonNet()
    x = ms.Tensor(np.arange(4).reshape((2, 2)).astype(np.float32))
    output = net(x, 0, False, 9, ms.Tensor([False, True]), False)
    expect_output = np.array([9., 1.], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)

    # test 2: using default args.
    output = net(x)
    assert np.allclose(output.asnumpy(), 0.0)

    # test 3: using k-v args.
    output = net(x, axis=0, keepdims=False, initial=9, where=ms.Tensor([False, True]), return_indices=False)
    expect_output = np.array([9., 1.], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_method_min_pyboost(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.min.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = MinPyboostNet()
    x = ms.Tensor(np.arange(4).reshape((2, 2)).astype(np.float32))
    output = net(x)
    assert np.allclose(output.asnumpy(), 0.0)
