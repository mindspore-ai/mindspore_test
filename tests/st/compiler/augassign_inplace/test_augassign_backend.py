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
from mindspore import Tensor, jit
from mindspore import Parameter
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_augassign_backend():
    """
    Feature: Support augassign inplace in kbk mode.
    Description: Support augassign inplace in kbk mode.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_zero = Parameter(Tensor(0, mstype.float32), name='zero')
            self.param_a = Parameter(Tensor(15, mstype.float32), name='a')

        def construct(self):
            out0 = self.param_zero
            out1 = self.param_a

            out1 += self.param_a
            out0 += self.param_a
            return out0, out1

    pynative_output = Net()()

    net0 = Net()
    net0.construct = jit(net0.construct, backend='GE')
    graph_output_ge = net0()

    net1 = Net()
    net1.construct = jit(net1.construct, backend='ms_backend')
    graph_output = net1()

    assert graph_output_ge == pynative_output
    assert graph_output == pynative_output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_initial_scalar_body_tensor1():
    """
    Feature: While specialize.
    Description: Test scalar arg when first entry of while and set to tensor in body.
    Expectation: No exception in infer process.
    """

    def func(x, a, b):
        y = 1
        while a < b:
            while a < b - 1:
                y = Tensor(2, ms.float32)
                a += 1
            a += 1
        return x + y

    @jit(backend='ms_backend')
    def test_net(x, a, b):
        out = x
        while a < b:
            while a < b - 1:
                out = func(out, a, b)
                a += 1
            a += 1
        return out

    input_np_x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    input_me_a = Tensor(2, ms.float32)
    input_me_b = Tensor(6, ms.float32)
    test_net(input_me_x, input_me_a, input_me_b)
