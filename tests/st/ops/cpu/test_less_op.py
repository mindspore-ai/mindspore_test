# Copyright 2020 Huawei Technologies Co., Ltd
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
import mindspore.context as context
import mindspore.nn as nn
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.ops = P.Less()

    def construct(self, x, y):
        return self.ops(x, y)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('dtype', [np.int32, np.int64, np.float32, np.float64])
def test_net(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Less
    Expectation: the result match to numpy
    """
    x0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(dtype)
    y0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(dtype)
    x1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(dtype)
    y1_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(dtype)
    x2_np = np.random.randint(1, 5, (2, 1, 1, 4)).astype(dtype)
    y2_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(dtype)
    x3_np = np.random.randint(1, 5, 1).astype(dtype)
    y3_np = np.random.randint(1, 5, 1).astype(dtype)
    x4_np = np.array(768).astype(dtype)
    y4_np = np.array(3072.5).astype(dtype)

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)
    x4 = Tensor(x4_np)
    y4 = Tensor(y4_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = Net()
    out = net(x0, y0).asnumpy()
    expect = x0_np < y0_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x1, y1).asnumpy()
    expect = x1_np < y1_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x2, y2).asnumpy()
    expect = x2_np < y2_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x3, y3).asnumpy()
    expect = x3_np < y3_np
    assert np.all(out == expect)
    assert out.shape == expect.shape

    out = net(x4, y4).asnumpy()
    expect = x4_np < y4_np
    assert np.all(out == expect)
    assert out.shape == expect.shape
