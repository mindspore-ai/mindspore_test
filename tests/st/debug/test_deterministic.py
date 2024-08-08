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

from mindspore import ops, nn, context
import mindspore as ms

class ReduceNet(nn.Cell):
    def __init__(self):
        super(ReduceNet, self).__init__()
        self.reducesum = ops.ReduceSum(keep_dims=False)

    def construct(self, x):
        output = self.reducesum(x, 1)
        return output


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level0', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_deterministic(mode):
    context.set_context(mode=mode, deterministic="ON")
    x = ms.Tensor(np.random.randn(16, 1024), ms.float32)
    reduce_net = ReduceNet()
    output1 = reduce_net(x)
    output2 = reduce_net(x)
    assert np.allclose(output1.asnumpy(), output2.asnumpy(), rtol=0, atol=0)
