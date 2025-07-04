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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
    def construct(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_net():
    x = np.arange(1 * 1 * 6 * 6).reshape((1, 1, 6, 6)).astype(np.float32)
    net = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='valid')
    out = net(Tensor(x))

    out_shape = out.asnumpy().shape
    sens = np.arange(int(np.prod(out_shape))).reshape(out_shape).astype(np.float32)
    backword_net = Grad(net)
    output = backword_net(Tensor(x), Tensor(sens))
    print(len(output))
    print(output[0].asnumpy())


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('input_dtype', [np.float16, np.float64])
def test_net_dtype(input_dtype):
    """
    Feature: test avg_pool_grad op.
    Description: backward.
    Expectation: expect correct backward result.
    """
    x = np.arange(1 * 1 * 6 * 6).reshape((1, 1, 6, 6)).astype(input_dtype)
    net = nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='valid')
    out = net(Tensor(x))

    out_shape = out.asnumpy().shape
    sens = np.arange(int(np.prod(out_shape))).reshape(out_shape).astype(input_dtype)
    backword_net = Grad(net)
    output = backword_net(Tensor(x), Tensor(sens))
    expect_output = np.array([[[[0., 0., 0.1111, 0.1111, 0.1111, 0.],
                                [0., 0., 0.1111, 0.1111, 0.1111, 0.],
                                [0.2222, 0.2222, 0.6665, 0.4443, 0.4443, 0.],
                                [0.2222, 0.2222, 0.5557, 0.3333, 0.3333, 0.],
                                [0.2222, 0.2222, 0.5557, 0.3333, 0.3333, 0.],
                                [0., 0., 0., 0., 0., 0.]]]]).astype(input_dtype)
    assert np.allclose(output[0].asnumpy(), expect_output, 1e-3, 1e-3)
