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
import mindspore as ms
from mindspore import nn, mint
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_ref_key():
    """
    Feature: Test parameter.
    Description: Parameters with different ref_key.
    Expectation: No exception.
    """
    def Normalize(in_channels, num_groups=32):
        return mint.nn.GroupNorm(num_groups=num_groups, num_channels=in_channels).to_float(ms.float32)

    class Block(nn.Cell):
        def __init__(self, in_channels=512, out_channels=512):
            super().__init__()
            self.norm1 = Normalize(in_channels)
            self.norm2 = Normalize(out_channels)
            self.norm1.weight = ms.Parameter(ms.ops.ones_like(self.norm1.weight))
            self.norm2.weight = ms.Parameter(ms.ops.ones_like(self.norm2.weight) * 2)

        def construct(self, h):
            w1 = self.norm1.weight.value()
            w2 = self.norm2.weight.value()
            h1 = self.norm1(h)
            h2 = self.norm2(h1)
            return h1, h2, w1, w2

    ms.set_context(mode=ms.GRAPH_MODE)
    x = ms.Tensor(np.random.randn(1, 512, 32, 32), ms.float32)
    net = Block()
    [_, _, w1, w2] = net(x)
    assert np.allclose(w1 * 2, w2)
