# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P


class AssignAdd(nn.Cell):
    def __init__(self, value):
        super(AssignAdd, self).__init__()
        self.var = Parameter(value, name="var")
        self.add = P.AssignAdd()

    def construct(self, y):
        self.add(self.var, y)
        return self.var


def get_output(x2, y2, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    if enable_graph_kernel:
        context.set_context(graph_kernel_flags="--enable_expand_ops=AssignAdd")
    add = AssignAdd(x2)
    result_gk_on_1 = add(y2)
    add_2 = AssignAdd(result_gk_on_1)
    result_gk_on_2 = add_2(y2)
    output = [result_gk_on_1, result_gk_on_2]
    return output


def assign_add():
    x2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float32))
    y2 = Tensor(np.arange(1 * 3 * 3 * 3).reshape(1, 3, 3, 3).astype(np.float16))

    expect = get_output(x2, y2, False)
    output = get_output(x2, y2, True)
    e1, e2 = list(expect)
    o1, o2 = list(output)

    assert np.allclose(o1.asnumpy(), e1.asnumpy())
    assert np.allclose(o2.asnumpy(), e2.asnumpy())


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_assign_add_gpu():
    """
    Feature: test graph kernel AssignAdd
    Description: run test case on GPU
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    assign_add()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_assign_add_ascend():
    """
    Feature: test graph kernel AssignAdd
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE)
    assign_add()
