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
import mindspore.context as context
import mindspore.nn as nn
import numpy as np
import pytest
from scipy import special

import mindspore as ms
from mindspore import Tensor
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


class NetErf(nn.Cell):
    def __init__(self):
        super(NetErf, self).__init__()
        self.erf = P.Erf()

    def construct(self, x):
        return self.erf(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_erf_dshape():
    """
    Feature: Test erf dynamic shape.
    Description: Test erf dynamic shape.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = NetErf()
    input_x_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    net.set_inputs(input_x_dyn)
    input_x = Tensor(np.random.random(([3, 10])), dtype=ms.float32)
    output = net(input_x)
    expect_shape = (3, 10)
    assert output.asnumpy().shape == expect_shape


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(2,), (4, 5), (3, 4, 5, 6)])
@pytest.mark.parametrize('dtype, tol', [(np.float16, 1.0e-3), (np.float32, 1.0e-4)])
def test_erf(mode, shape, dtype, tol):
    """
    Feature: ALL To ALL
    Description: test cases for erf
    Expectation: the result match to scipy
    """
    context.set_context(mode=mode, device_target="CPU")
    erf = P.Erf()
    input_x = np.random.randn(*shape).astype(dtype)
    output = erf(Tensor(input_x))
    expect_output = special.erf(input_x)
    diff = output.asnumpy() - expect_output
    error = np.ones(shape=expect_output.shape) * tol
    assert np.all(np.abs(diff) < error)
