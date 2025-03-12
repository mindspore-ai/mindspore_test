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

import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor

class Net(nn.Cell):
    def construct(self, x, shape):
        return x.broadcast_to(shape)

@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_broadcast_to(mode):
    """
    Feature: tensor.broadcast_to
    Description: Verify the result of tensor.broadcast_to
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    shape = (4, 5, 2, 3, 4, 5, 6)
    x_np = np.random.rand(2, 3, 1, 5, 1).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    output = net(x, shape)
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (3, 5, 7, 4, 5, 6)
    x_np = np.arange(20).reshape((4, 5, 1)).astype(np.int32)
    x = Tensor(x_np, ms.int32)
    output = net(x, shape)
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (8, 5, 7, 4, 5, 6)
    x_np = np.arange(24).reshape((1, 4, 1, 6)).astype(np.bool_)
    x = Tensor(x_np, ms.bool_)
    output = net(x, shape)
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (3, 4, 5, 2, 3, 4, 5, 7)
    x_np = np.random.rand(2, 3, 1, 5, 1).astype(np.float16)
    x = Tensor(x_np, ms.float16)
    output = net(x, shape)
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (3, 4, 5, 6)
    x_np = np.random.rand(3, 1, 5, 1).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    output = net(x, shape)
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (2, 3, 4, 5)
    x_np = np.random.rand(4, 5).astype(np.float32)
    x = Tensor(x_np, ms.float32)
    output = net(x, shape)
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect)
