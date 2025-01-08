# Copyright 2024 Huawei Technologies Co., Ltd
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
import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def construct(self, x):
        return x.t()

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

@arg_mark(
    plat_marks=['platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_t(mode):
    """
    Feature: Tensor.t
    Description: Verify the result of Tensor.t
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = ms.Tensor(generate_random_input((2, 3), np.float32))
    net = Net()
    output = net(x)
    expect_output = np.transpose(x)
    assert np.allclose(output.asnumpy(), expect_output)
