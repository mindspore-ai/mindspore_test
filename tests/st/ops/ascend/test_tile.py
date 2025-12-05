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
"""test for tile"""
from tests.mark_utils import arg_mark
import pytest
import numpy as np

from mindspore import context
from mindspore import nn
from mindspore import Tensor
from mindspore.ops import function as F
import mindspore as ms

context.set_context(device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.tile = F.tile
        self.x = Tensor(np.ones((1, 1, 3, 4)), dtype=ms.bfloat16)
        self.multiples = (2, 2, 1, 1)

    def construct(self):
        return self.tile(self.x, self.multiples)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_tile_bfloat16(mode):
    """
    Feature: test Tile forward.
    Description: test bfloat16 inputs.
    Expectation: compare the result with exception value.
    """
    context.set_context(mode=mode)
    net = Net()
    output = net()
    expect_output = np.ones((2, 2, 3, 4)).astype(np.float32)
    np.testing.assert_allclose(output.float().asnumpy(), expect_output, 1e-3, 1e-3)
