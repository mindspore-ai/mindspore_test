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
""" test graph fallback control flow if after for scenario"""
from mindspore import Tensor, jit, context
from tests.st.compiler.fallback.cases_register import case_register

context.set_context(mode=context.GRAPH_MODE)


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_for_tensor():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_after_for():
        x = Tensor(7)
        y = Tensor(0)
        for _ in range(3):
            x += 1
            y += 3
        if x > y:
            return x + y
        return x - y
    res = control_flow_if_after_for()
    assert res == 19


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_for_tensor_2():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_after_for():
        x = Tensor(7)
        y = Tensor(0)
        for i in range(3):
            x += i
            y += 3 * i
        if x == y:
            return x + y
        return x - y
    res = control_flow_if_after_for()
    assert res == 1


@case_register.level1
@case_register.target_gpu
@case_register.target_ascend
def test_if_after_for_tensor_3():
    """
    Feature: JIT Fallback
    Description: Test fallback with control flow.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def control_flow_if_after_for():
        x = Tensor(7)
        y = Tensor(10)
        for i in range(3):
            x += i
        if x == y:
            return x + y
        return x - y
    res = control_flow_if_after_for()
    assert res == 20
