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
def test_for_after_for_in_if_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def func3303():
        x = np.array([1, 2, 3])
        y = np.array([5, 6, 7])
        k = []
        if x[2] < y[0]:
            y = y - x
            for i in y:
                k.append(i)

        z = Tensor(k)
        out = 1
        for i in z:
            out = out * i
        return out

    res = func3303()
    assert res == 64


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_for_after_for_in_if_4():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """

    @jit(backend="ms_backend")
    def func3304():
        x = Tensor([1])
        y = Tensor([2])
        if max(x, y) == Tensor([1]) or min(x, y) == Tensor([2]):
            return x

        z = (Tensor(1), Tensor(2), Tensor(3))
        for i in zip(z):
            x = x * i
        return x

    res = func3304()
    assert res == 6
