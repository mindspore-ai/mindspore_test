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

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

class Net(nn.Cell):
    def construct(self, x, mat2):
        return x.bmm(mat2)

@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_bmm(mode):
    """
    Feature: tensor.bmm
    Description: Verify the result of tensor.bmm
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x = generate_random_input((1, 3, 3), np.float32)
    y = generate_random_input((1, 3, 3), np.float32)
    expect = x @ y
    output = net(Tensor(x), Tensor(y))
    assert np.allclose(output.asnumpy(), expect)
