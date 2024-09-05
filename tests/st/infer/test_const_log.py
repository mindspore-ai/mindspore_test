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
"""test GE const"""

import numpy as np
from mindspore import context, nn, Tensor, Parameter

class Net(nn.Cell):
    def __init__(self, parameter_data):
        super().__init__()
        self.parameter = Parameter(Tensor(parameter_data), name="my_parameter")

    def construct(self, x):
        return x + self.parameter
shape = (16, 512)
np_data = np.ones(shape).astype(np.float32)
x_data = np.ones(shape).astype(np.float32)

context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O2"})
net = Net(np_data)
net.phase = "prefill"
out = net(Tensor(x_data))

np_out = np_data + x_data
assert np.allclose(out.asnumpy(), np_out)
