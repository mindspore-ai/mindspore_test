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


class MeanNet(nn.Cell):
    def construct(self, x, axis=None, keep_dims=False):
        return x.mean(axis, keep_dims)


class MeanKVNet(nn.Cell):
    def construct(self, x, axis=None, keep_dims=False):
        return x.mean(axis=axis, keep_dims=keep_dims)


class MeanKVDisruptNet(nn.Cell):
    def construct(self, x, axis=None, keep_dims=False):
        return x.mean(keep_dims=keep_dims, axis=axis)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_method_mean(mode):
    """
    Feature: Functional.
    Description: Test functional feature with Tensor.mean.
    Expectation: Run success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})

    # test 1: using positional args
    net = MeanNet()
    x = ms.Tensor(np.random.randn(3, 4, 5, 6).astype(np.float32))
    output = net(x, 1, True)
    result = output.shape
    expected = np.array([3, 1, 5, 6], dtype=np.float32)
    assert np.allclose(result, expected)

    # test 2: using default args.
    x = ms.Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
                            [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
                            [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]), ms.float32)
    output = net(x)
    expected = 5.0
    assert np.allclose(output.asnumpy(), expected)

    # test 3: using k-v args.
    net = MeanKVNet()
    output = net(x, axis=0, keep_dims=True)
    expected = np.array([[[4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
                          [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
                          [6.0, 6.0, 6.0, 6.0, 6.0, 6.0]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)

    # test 4: using k-v out of order args.
    net = MeanKVDisruptNet()
    output = net(x, axis=1, keep_dims=True)
    expected = np.array([[[2.0, 2.0, 2.0, 2.0, 2.0, 2.0]],
                         [[5.0, 5.0, 5.0, 5.0, 5.0, 5.0]],
                         [[8.0, 8.0, 8.0, 8.0, 8.0, 8.0]]], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expected)
