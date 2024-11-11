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
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import numpy as np
import pytest

from tests.mark_utils import arg_mark


class IsCloseNet(nn.Cell):
    def construct(self, x, x2, rtol=1e-05, atol=1e-08, equal_nan=True):
        return x.isclose(x2, rtol, atol, equal_nan)


class IsCloseKVNet(nn.Cell):
    def construct(self, x, x2, rtol=1e-05, atol=1e-08, equal_nan=True):
        return x.isclose(x2, rtol=rtol, atol=atol, equal_nan=equal_nan)


class IsCloseKVDisruptNet(nn.Cell):
    def construct(self, x, x2, rtol=1e-05, atol=1e-08, equal_nan=True):
        return x.isclose(x2, equal_nan=equal_nan, atol=atol, rtol=rtol)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_isclose(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.isclose.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    # test 1: using positional args
    net = IsCloseNet()
    x = ms.Tensor([1.3, 2.1, 3.2, 4.1, 5.1, 6.1], dtype=mstype.float16)
    y = ms.Tensor([1.3, 3.3, 2.3, 3.1, 5.1, 5.6], dtype=mstype.float16)
    output = net(x, y, 1e-05, 1e-08, True)
    expected = np.array([True, False, False, False, True, False], dtype=np.bool_)
    assert np.allclose(output.asnumpy(), expected)

    # test 2: using default args
    net = IsCloseNet()
    x = ms.Tensor([1.3, 2.1, 3.2, 4.1, 5.1, 6.1], dtype=mstype.float16)
    y = ms.Tensor([1.3, 3.3, 2.3, 3.1, 5.1, 5.6], dtype=mstype.float16)
    output = net(x, y)
    expected = np.array([True, False, False, False, True, False], dtype=np.bool_)
    assert np.allclose(output.asnumpy(), expected)

    # test 3: using k-v args
    net = IsCloseKVNet()
    x = ms.Tensor([1.3, 2.1, 3.2, 4.1, 5.1, 6.1], dtype=mstype.float16)
    y = ms.Tensor([1.3, 3.3, 2.3, 3.1, 5.1, 5.6], dtype=mstype.float16)
    output = net(x, y, rtol=1e-03, atol=1e-04, equal_nan=False)
    expected = np.array([True, False, False, False, True, False], dtype=np.bool_)
    assert np.allclose(output.asnumpy(), expected)

    # test 3: using k-v args
    net = IsCloseKVDisruptNet()
    x = ms.Tensor([1.3, 2.1, 3.2, 4.1, 5.1, 6.1], dtype=mstype.float16)
    y = ms.Tensor([1.3, 3.3, 2.3, 3.1, 5.1, 5.6], dtype=mstype.float16)
    output = net(x, y, rtol=1e-03, atol=1e-04, equal_nan=False)
    expected = np.array([True, False, False, False, True, False], dtype=np.bool_)
    assert np.allclose(output.asnumpy(), expected)
