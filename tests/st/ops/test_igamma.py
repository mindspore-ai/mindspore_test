# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import pytest
import numpy as np

from mindspore import Tensor
import mindspore.context as context
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_igamma_functional_api_modes(mode):
    """
    Feature: Test igamma functional api.
    Description: Test igamma functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode)
    a = Tensor([2.0, 4.0, 6.0, 8.0], mstype.float32)
    x = Tensor([2.0, 3.0, 4.0, 5.0], mstype.float32)
    output = F.igamma(a, x)
    expected = np.array([0.593994, 0.35276785, 0.21486944, 0.13337152])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_igamma_functional_api_compile(mode):
    """
    Feature: Test igamma functional api.
    Description: Test igamma functional api for Graph and PyNative modes with scalar input.
    Expectation: Network compile succeed.
    """
    context.set_context(mode=mode)
    a = Tensor(np.random.uniform(0, 15, (3,)), mstype.float64)
    x = Tensor(np.random.uniform(0, 15, (1,)), mstype.float64)
    F.igamma(a, x).asnumpy()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_igamma_tensor_api_modes(mode):
    """
    Feature: Test igamma tensor api.
    Description: Test igamma tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    a = Tensor([2.0, 4.0, 6.0, 8.0], mstype.float32)
    x = Tensor([2.0, 3.0, 4.0, 5.0], mstype.float32)
    output = a.igamma(x)
    expected = np.array([0.593994, 0.35276785, 0.21486944, 0.13337152])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_igammac_functional_api_modes(mode):
    """
    Feature: Test igamma functional api.
    Description: Test igamma functional api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode)
    a = Tensor([2.0, 4.0, 6.0, 8.0], mstype.float32)
    x = Tensor([2.0, 3.0, 4.0, 5.0], mstype.float32)
    output = F.igammac(a, x)
    expected = np.array([0.40600586, 0.6472318, 0.7851304, 0.8666283])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_igammac_tensor_api_modes(mode):
    """
    Feature: Test igamma tensor api.
    Description: Test igamma tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    a = Tensor([2.0, 4.0, 6.0, 8.0], mstype.float32)
    x = Tensor([2.0, 3.0, 4.0, 5.0], mstype.float32)
    output = a.igammac(x)
    expected = np.array([0.40600586, 0.6472318, 0.7851304, 0.8666283])
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)
