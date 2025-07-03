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

from mindspore import nn, Tensor, jit
from mindspore import dtype as mstype
import numpy as np


class DynamicShapeNet3(nn.Cell):
    def __init__(self):
        super().__init__()
        self.num = 2

    @jit(dynamic=1)
    def construct(self, x, y):
        return x + self.num, y

x1 = Tensor(np.random.rand(2, 3), mstype.float32)
x2 = Tensor(np.random.rand(2, 4), mstype.float32)
x3 = Tensor(np.random.rand(2, 5), mstype.float32)
x4 = Tensor(np.random.rand(2, 6), mstype.float32)

net = DynamicShapeNet3()
output1 = net(x1, 1.0)
output2 = net(x2, 1.0)
output3 = net(x3, 1.0)
output4 = net(x4, 1.0)
