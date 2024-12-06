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
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, axis=(), keep_dims=False):
        return x.any(axis, keep_dims)


class Net2(nn.Cell):
    def construct(self, x, dim=None, keepdim=False):
        return x.any(dim=dim, keepdim=keepdim)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_any(mode):
    """
    Feature: tensor.any
    Description: Verify the result of any
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    net2 = Net2()

    x = Tensor(np.array([[True, False], [False, False]]))
    output = net(x, keep_dims=True)
    output_2 = net2(x, keepdim=True)
    assert np.allclose(output.asnumpy(), np.array([True]))
    assert np.allclose(output.shape, (1, 1))
    assert np.allclose(output_2.asnumpy(), np.array([True]))
    assert np.allclose(output_2.shape, (1, 1))

    output2 = net(x, axis=0)
    output2_2 = net2(x, dim=0)
    assert np.allclose(output2.asnumpy(), np.array([True, False]))
    assert np.allclose(output2.shape, (2,))
    assert np.allclose(output2_2.asnumpy(), np.array([True, False]))
    assert np.allclose(output2_2.shape, (2,))

    output3 = net(x, axis=1)
    output3_2 = net2(x, dim=1)
    assert np.allclose(output3.asnumpy(), np.array([True, False]))
    assert np.allclose(output3.shape, (2,))
    assert np.allclose(output3_2.asnumpy(), np.array([True, False]))
    assert np.allclose(output3_2.shape, (2,))
