# Copyright 2021 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops.operations import _grad_ops as G

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetACosGrad(nn.Cell):
    def __init__(self):
        super(NetACosGrad, self).__init__()
        self.acosGrad = G.ACosGrad()

    def construct(self, x, dy):
        return self.acosGrad(x, dy)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_acos_grad():
    x = np.array([-0.5, 0, 0.5]).astype('float32')
    dy = np.array([1, 0, -1]).astype('float32')
    acos_grad = NetACosGrad()
    output = acos_grad(Tensor(x), Tensor(dy))
    print(output)
    expect = -dy / np.sqrt(1 - x * x)
    assert np.allclose(output.asnumpy(), expect)
