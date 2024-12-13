# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from tests.mark_utils import arg_mark


class Net(nn.Cell):

    def construct(self, x, dim, start, length):
        return x.narrow(dim, start, length)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_narrow(mode):
    """
    Feature: tensor.narrow
    Description: Verify the result of tensor.narrow
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], ms.int32)
    net = Net()
    output1 = net(x, 0, 0, 2)
    expect_x1 = Tensor([[1, 2, 3], [4, 5, 6]], ms.int32)
    assert np.allclose(output1.asnumpy(), expect_x1.asnumpy())

    output2 = net(x, 1, 1, 2)
    expect_x2 = Tensor([[2, 3], [5, 6], [8, 9]], ms.int32)

    assert np.allclose(output2.asnumpy(), expect_x2.asnumpy())
