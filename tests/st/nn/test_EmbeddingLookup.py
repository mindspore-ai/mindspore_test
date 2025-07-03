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
    def __init__(self):
        super(Net, self).__init__()
        self.embeddinglookup = nn.EmbeddingLookup(4, 2, dtype=ms.float32)

    def construct(self, x):
        out = self.embeddinglookup(x)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_embeddinglookup_para_customed_dtype(mode):
    """
    Feature: EmbeddingLookup
    Description: Verify the result of EmbeddingLookup specifying customed para dtype.
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": "O0"})
    net = Net()
    x = Tensor(np.array([[1, 0], [3, 2]]), ms.int32)
    output = net(x)
    expect_output_shape = (2, 2, 2)
    assert np.allclose(expect_output_shape, output.shape)
