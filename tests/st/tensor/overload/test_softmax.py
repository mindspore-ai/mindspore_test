# Copyright 2025 Huawei Technologies Co., Ltd
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

import pytest
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def construct(self, x, dim, dtype=None):
        return x.softmax(dim, dtype=dtype)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_tensor_softmax(mode):
    """
    Feature: Tensor.softmax
    Description: Verify the result of Tensor.softmax
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    net = Net()
    x_np = np.array([1., 2., 3., 4., 5.]).astype(np.float32)
    expect_out = np.array([0.0117, 0.0317, 0.0861, 0.2341, 0.6364]).astype(np.float16)
    output = net(Tensor(x_np), 0, dtype=ms.float16)
    assert np.allclose(output.asnumpy(), expect_out, atol=1e-4, rtol=1e-4)

    x_np = np.array([7., 7., 7., 8., 9.]).astype(np.float32)
    expect_out = np.array([0.0763, 0.0763, 0.0763, 0.2074, 0.5637]).astype(np.float32)
    output = net(Tensor(x_np), 0)
    assert np.allclose(output.asnumpy(), expect_out, atol=1e-4, rtol=1e-4)
