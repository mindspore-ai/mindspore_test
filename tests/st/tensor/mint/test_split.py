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
import os
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn

class SplitNet(nn.Cell):
    def construct(self, x, split_size_or_sections, axis=0):
        out = x.split(split_size_or_sections, axis)
        return out


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_f_split_int(mode):
    """
    Feature: split
    Description: Verify the result of split.
    Expectation: success
    """
    os.environ["MS_TENSOR_API_ENABLE_MINT"] = '1'
    ms.set_context(mode=mode)
    net = SplitNet()
    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    split_size_or_sections = 5
    out = net(x, split_size_or_sections)
    expect = [np.array(np.arange(10).reshape((5, 2)), dtype=np.float32),
              np.array(np.arange(10, 20).reshape((5, 2)), dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)

    split_size_or_sections = 1
    out = net(x, split_size_or_sections, axis=1)
    expect = [np.array(np.arange(0, 20, 2).reshape((10, 1)), dtype=np.float32),
              np.array(np.arange(1, 20, 2).reshape((10, 1)), dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)
    del os.environ["MS_TENSOR_API_ENABLE_MINT"]


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_f_split_list(mode):
    """
    Feature: split
    Description: Verify the result of split.
    Expectation: success
    """
    os.environ["MS_TENSOR_API_ENABLE_MINT"] = '1'
    ms.set_context(mode=mode)
    net = SplitNet()
    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    split_size_or_sections = [2, 3, 5]
    out = net(x, split_size_or_sections)
    expect = [np.array([[0, 1], [2, 3]], dtype=np.float32),
              np.array([[4, 5], [6, 7], [8, 9]], dtype=np.float32),
              np.array([[10, 11], [12, 13], [14, 15], [16, 17], [18, 19]], dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)

    a = np.array(np.arange(20).reshape((2, 10)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    out = net(x, split_size_or_sections, axis=1)
    expect = [np.array([[0, 1], [10, 11]], dtype=np.float32),
              np.array([[2, 3, 4], [12, 13, 14]], dtype=np.float32),
              np.array([[5, 6, 7, 8, 9], [15, 16, 17, 18, 19]], dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)
    del os.environ["MS_TENSOR_API_ENABLE_MINT"]
