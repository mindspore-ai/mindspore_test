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
import pytest
from tests.mark_utils import arg_mark
import numpy as np

import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
from mindspore.common import dtype as mstype


class CholeskyTensorNet(nn.Cell):
    def construct(self, x):
        return x.cholesky(upper=False)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tensor_cholesky(mode):
    """
    Feature: Test cholesky tensor api.
    Description: Test cholesky tensor api for Graph and PyNative modes.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    x = Tensor([[1.0, 1.0], [1.0, 2.0]], mstype.float32)
    net = CholeskyTensorNet()
    output = net(x)
    expect_output = np.array([[1., 0.], [1., 1.]])
    assert np.allclose(output.asnumpy(), expect_output)
