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
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, jit
from mindspore import dtype as mstype
from mindspore.common import Parameter

steps = 30

ms.set_context(mode=ms.PYNATIVE_MODE)

affinity_cpu_list = ["0-10", "21-30"]
module_to_cpu_dict = {"main": [0, 1, 2, 3], "minddata": [4, 5], "other": [6, 7],
                      "runtime": [8, 9], "pynative": [10, 11, 21, 100]}
ms.runtime.set_cpu_affinity(True, affinity_cpu_list, module_to_cpu_dict)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.mul = P.Mul()

    @jit(backend="ms_backend")
    def construct(self, x):
        x = self.add(x, self.param)
        for _ in range(5):
            x = self.add(x, 0.1)
            x = self.add(x, 0.2)
        x = self.mul(x, 2)
        x = self.add(x, 0.5)
        return x

base_shape = (2, 3)
net = Net()
dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
net.set_inputs(dyn_input_data)

for i in range(steps):
    input_data = Tensor(np.full(base_shape, i).astype(np.float32))
    output = net(input_data)
