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

def generate_expect_forward_output(input_x, vec1, vec2, beta=1, alpha=1):
    return beta * input_x + alpha * np.outer(vec1, vec2)

class Net(nn.Cell):
    def construct(self, input_x, vec1, vec2, beta=1, alpha=1):
        return input_x.addr(vec1, vec2, beta=beta, alpha=alpha)

@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_addr(mode):
    """
    Feature: tensor.addr
    Description: Verify the result of tensor.addr
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    net = Net()
    x_np = generate_random_input((6, 3), np.float32)
    input_x = Tensor(x_np, ms.float32)
    vec1_np = generate_random_input((6,), np.float32)
    vec1 = Tensor(vec1_np, ms.float32)
    vec2_np = generate_random_input((3,), np.float32)
    vec2 = Tensor(vec2_np, ms.float32)
    beta = 1.0
    alpha = 1.0
    output = net(input_x, vec1, vec2, beta=beta, alpha=alpha)
    expected = generate_expect_forward_output(x_np, vec1_np, vec2_np, beta=beta, alpha=alpha)
    assert np.allclose(output.asnumpy(), expected)
