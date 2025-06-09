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

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.runtime as rt
from mindspore import Tensor, context
from mindspore.common import Parameter
from mindspore import dtype as mstype
from tests.mark_utils import arg_mark

context.set_context(
    mode=context.GRAPH_MODE,
    jit_config={
        "jit_level": "O0",
        "infer_boost": "on"
    },
    max_call_depth=600000
)

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.mul = P.Mul()

    def construct(self, x):
        for _ in range(5):
            x = self.add(x, 0.1)
            x = self.add(x, 0.2)
        x = self.mul(x, 2)
        x = self.add(x, 0.5)
        return x

class SeqNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = Net()

    def construct(self, x):
        output = self.net(x)
        return output

@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='essential'
)
def test_dynamic_shape():
    """
    Feature: graph mode support capture graph
    Description: Test dynamic shape scene for capture graph
    Expectation: No exception and result is correct
    """
    rt.set_kernel_launch_capture(True)
    input_data = Tensor(np.ones((2, 3)).astype(np.float32))
    new_input1 = Tensor(np.ones((2, 5)).astype(np.float32))
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)

    net = SeqNet()
    net.set_inputs(dyn_input_data)
    net.phase = "increment"

    expected_results = [
        np.ones((2, 3), dtype=np.float32) * 5.5,
        np.ones((2, 3), dtype=np.float32) * 5.5,
        np.ones((2, 5), dtype=np.float32) * 5.5,
        np.ones((2, 5), dtype=np.float32) * 5.5,
        np.ones((2, 5), dtype=np.float32) * 5.5,
    ]

    for i in range(5):
        if i < 2:
            output = net(input_data)
        else:
            output = net(new_input1)
        assert np.allclose(output.asnumpy(), expected_results[i], 0.0001, 0.0001)
 