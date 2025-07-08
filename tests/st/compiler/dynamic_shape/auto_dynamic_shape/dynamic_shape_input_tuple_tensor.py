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

from mindspore import nn, Tensor, jit, mutable
from mindspore import dtype as mstype
import numpy as np


class DynamicShapeNet2(nn.Cell):
    def __init__(self):
        super().__init__()
        self.num = 0

    @jit(dynamic=1)
    def construct(self, x):
        out = x[self.num] + x[1]
        return out

x1 = Tensor(np.random.rand(2, 3, 4), mstype.float32)
y1 = Tensor(np.random.rand(2, 3, 4), mstype.float32)
tuple1 = (x1, y1)
x2 = Tensor(np.random.rand(3, 3, 4), mstype.float32)
y2 = Tensor(np.random.rand(3, 3, 4), mstype.float32)
tuple2 = (x2, y2)
x3 = Tensor(np.random.rand(4, 3, 4), mstype.float32)
y3 = Tensor(np.random.rand(4, 3, 4), mstype.float32)
tuple3 = (x3, y3)
x4 = Tensor(np.random.rand(5, 3, 4), mstype.float32)
y4 = Tensor(np.random.rand(5, 3, 4), mstype.float32)
tuple4 = (x4, y4)

net = DynamicShapeNet2()
output1 = net(mutable(tuple1))
output2 = net(mutable(tuple2))
output3 = net(mutable(tuple3))
output4 = net(mutable(tuple4))
