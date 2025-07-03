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
def test_if_in_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for():
        x = Tensor(7)
        y = Tensor(0)
        for _ in range(3):
            if y < Tensor(10):
                y += x
        return y
    res = control_flow_for()
    assert res == 14


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_if_in_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for():
        x = Tensor(7)
        y = Tensor(0)
        for _ in range(3):
            if y < Tensor(10):
                y += x
            else:
                y += Tensor(5)
        return y
    res = control_flow_for()
    assert res == 19


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_if_in_for_tensor_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for():
        x = Tensor(7)
        y = Tensor(0)
        for _ in range(3):
            if y < Tensor(10):
                y += x
            y += Tensor(1)
        return y
    res = control_flow_for()
    assert res == 17


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu',], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_if_in_for_numpy_5():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_for():
        x = np.array([1, 2, 3, 4])
        y = (Tensor(1), Tensor(3), Tensor(5))
        a = Tensor(0)
        for e in y:
            a += Tensor(sum(x)) + e
            if a <= 15:
                x += 1
        return a
    res = control_flow_for()
    assert res == 47
