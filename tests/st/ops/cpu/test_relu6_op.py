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
from mindspore.ops.operations import _grad_ops as G

from mindspore import Tensor
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetReLU6(nn.Cell):
    def __init__(self):
        super(NetReLU6, self).__init__()
        self.relu6 = P.ReLU6()

    def construct(self, x):
        return self.relu6(x)


class NetReLU6Grad(nn.Cell):
    def __init__(self):
        super(NetReLU6Grad, self).__init__()
        self.relu6_grad = G.ReLU6Grad()

    def construct(self, x, dy):
        return self.relu6_grad(dy, x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_relu6():
    """
    Feature: template
    Description: template
    Expectation: template
    """
    x = Tensor(np.array([[[[-1, 1, 10],
                           [5.9, 6.1, 6],
                           [10, 1, -1]]]]).astype(np.float32))
    expect = np.array([[[[0, 1, 6, ],
                         [5.9, 6, 6, ],
                         [6, 1, 0.]]]]).astype(np.float32)

    relu6 = NetReLU6()
    output = relu6(x)
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_relu6_grad():
    """
    Feature: template
    Description: template
    Expectation: template
    """
    x = Tensor(np.array([[[[-1, 1, 10],
                           [5.9, 6.1, 6],
                           [10, 1, -1]]]]).astype(np.float32))
    dy = Tensor(np.array([[[[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]]]]).astype(np.float32))
    expect = np.array([[[[0, 1, 0, ],
                         [1, 0, 1, ],
                         [0, 1, 0, ]]]]).astype(np.float32)
    error = np.ones(shape=[3, 3]) * 1.0e-6

    relu6_grad = NetReLU6Grad()
    output = relu6_grad(x, dy)
    diff = np.abs(output.asnumpy() - expect)
    assert np.all(np.abs(diff) < error)
