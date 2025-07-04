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
def test_while_after_while_in_if_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_while_after_while_in_if():
        x = Tensor([1])
        y = Tensor([2])
        z = Tensor([0])
        if z < x + y and x < y:
            y = y * x + Tensor([3])
            z = x + y
            while x * y > z:
                x = y - x
        while y > x:
            y -= x
            z = z + y
        return x, y, z
    res = control_flow_while_after_while_in_if()
    assert res == (1, 1, 16)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_while_after_while_in_if_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_while_after_while_in_if():
        x = Tensor([3])
        y = Tensor([5])
        z = Tensor([0])
        if z > x + y and x < y:
            y = y * x + Tensor([3])
            z = x + y
            while x * y > z:
                x = y - x
        elif x < y:
            while x * y > z:
                x = y - x
                z += 1
        else:
            x = x + y
        while y > x:
            y -= x
            z = z + y
        return x, y, z
    res = control_flow_while_after_while_in_if()
    assert res == (2, 1, 15)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_while_after_while_in_if_numpy_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_while_after_while_in_if():
        x = Tensor([1])
        y = Tensor([2])
        z = np.array([1]) + np.array([2])
        if Tensor(z) < x + y:
            y = y * x + Tensor([3])
            while np.array([5]) > z:
                z += 1
        else:
            y = y * x - Tensor(z)
        while y < x:
            y += (x + Tensor(z))
        return x, y, Tensor(z)
    res = control_flow_while_after_while_in_if()
    assert res == (1, 3, 3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_while_after_while_in_if_numpy():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_while_after_while_in_if():
        x = Tensor([1])
        y = Tensor([2])
        z = np.array([1]) + np.array([2])
        if Tensor(z) >= x + y:
            y = y * x + Tensor([3])
            while np.array([5]) > z:
                z += 1
        while y > x:
            y -= x
        return x, y, Tensor(z)
    res = control_flow_while_after_while_in_if()
    assert res == (1, 1, 5)
