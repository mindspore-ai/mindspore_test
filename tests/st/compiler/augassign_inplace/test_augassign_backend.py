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

import mindspore.nn as nn
from mindspore import context, Tensor, jit
from mindspore import Parameter
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_augassign_backend():
    """
    Feature: Support augassign inplace in kbk mode.
    Description: Support augassign inplace in kbk mode.
    Expectation: Run success.
    """
    net0 = Net()
    net0.construct = jit(net0.construct, backend='GE')
    graph_output_ge = net0()
    assert graph_output_ge[0] == Tensor(15, mstype.float32)
    assert graph_output_ge[1] == Tensor(30, mstype.float32)

    net1 = Net()
    net1.construct = jit(net1.construct, backend='ms_backend')
    graph_output = net1()
    assert graph_output[0] == Tensor(30, mstype.float32)
    assert graph_output[1] == Tensor(30, mstype.float32)

    context.set_context(mode=context.PYNATIVE_MODE)
    pynative_output = Net()()
    assert graph_output == pynative_output
