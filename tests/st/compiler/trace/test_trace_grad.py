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
""" test trace functions """

import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore.ops.functional import grad


@pytest.mark.skip(reason="core dump")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_trace_1():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            z = x + a
            z = z + self.x
            z = z * y
            return z

    trace_net = TraceNet()
    res1 = grad(trace_net)(ms.Tensor(1), ms.Tensor(3))
    res2 = grad(trace_net)(ms.Tensor(1), ms.Tensor(3))
    print(f'res1: {res1}, res2: {res2}')
    assert res1 == res2


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="No support")
def test_trace_2():
    """
    Feature: JIT trace function
    Description: JIT trace function
    Expectation: No exception
    """
    class TraceNet(ms.nn.Cell):
        def __init__(self):
            super(TraceNet, self).__init__()
            self.x = ms.Tensor(1)

        @ms.jit(capture_mode="trace")
        def construct(self, x, y):
            a = ms.Tensor(2)
            z = x + a
            z = z + self.x
            z = z * y
            return z

    class GradNet(ms.nn.Cell):
        def __init__(self):
            super(GradNet, self).__init__()
            self.net = TraceNet()

        def construct(self, x, y):
            z1 = x * y
            z2 = x + y
            z3 = self.net(z1, z2)
            return z3 * z3

    grad_net = GradNet()
    res1 = grad(grad_net)(ms.Tensor(1), ms.Tensor(3))
    res2 = grad(grad_net)(ms.Tensor(1), ms.Tensor(3))
    res3 = grad(grad_net)(ms.Tensor(1), ms.Tensor(3))
    res4 = grad(grad_net)(ms.Tensor(1), ms.Tensor(3))
    print(f'res1: {res1}, res2: {res2}, res3: {res3}, res4: {res4}')
    assert res1 == res2
    assert res2 == res3
    assert res3 == res4
