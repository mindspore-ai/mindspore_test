# Copyright 2022 Huawei Technologies Co., Ltd
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
""" test graph fallback control flow."""
import numpy as np
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_for_after_while_1():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def func():
        x = Tensor([0])
        y = Tensor([0])

        while x < Tensor([3]):
            x = x + Tensor([1])
        for _ in range(3):
            y = y - 1
        return x + y

    res = func()
    assert res == 0


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_for_after_while_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def func():
        x = Tensor([2])
        y = Tensor([2])
        while y < Tensor([5]) and x > Tensor([1]):
            y = y + Tensor([3])

        for i in range(3):
            z = Tensor([i])
            x = x + z

        return x, y

    res_x, res_y = func()
    assert res_x == 5
    assert res_y == 5


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_for_after_while_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def func():
        x = np.array([0])
        y = np.array([5, 6, 7])
        while x < 2:
            y = y - np.array([2, 3, 4])
            x = x + 2

        z = Tensor(y)
        out = 1
        for i in z:
            out = out * i
        return out

    res = func()
    assert res == 27
