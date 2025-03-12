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
import mindspore as ms
from mindspore import nn, Tensor
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x_np):
    return np.deg2rad(x_np)

class Net(nn.Cell):
    def construct(self, x):
        return x.deg2rad()


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_tensor_deg2rad(mode):
    """
    Feature: Tensor.deg2rad
    Description: Verify the result of Tensor.deg2rad
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_level='O0')

    net = Net()
    x_np = generate_random_input((3, 5), np.float32)
    output = net(Tensor(x_np))
    expected = generate_expect_forward_output(x_np)
    assert np.allclose(output.asnumpy(), expected)
