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
"""Test builtin function constant fold"""
import sys  
import pytest

from mindspore import Tensor
from mindspore import context
from mindspore.common.api import jit

from ..share.utils import match_array, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark

@pytest.fixture(autouse=True)  
def skip_if_python_version_too_high():  
    if sys.version_info >= (3, 11):  
        pytest.skip("Skipping tests on Python 3.11 and higher.") 

cfg = {"compile_with_try": False}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_abs():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn(x: Tensor):
        return abs(x) + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1, -1, 2, -2])
    o1 = fn(x)

    fn = jit(fn, mode="PIJit", jit_config=cfg)
    o2 = fn(x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_len():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    @jit(mode="PIJit", jit_config=cfg)
    def fn(x: Tensor):
        return len(x) + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1, 2, 3, 4])
    o = fn(x)

    assert o == 5
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_pow():
    """
    Feature: One stage basic operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn(x: Tensor):
        return pow(x, 2) + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1, -1, 2, -2])
    o1 = fn(x)

    fn = jit(fn, mode="PIJit", jit_config=cfg)
    o2 = fn(x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)
