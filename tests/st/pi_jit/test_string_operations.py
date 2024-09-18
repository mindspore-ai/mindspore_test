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
"""Test string operations"""
from mindspore import Tensor
from mindspore import context
from mindspore.common.api import jit
from .share.utils import match_array, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark


cfg = {
    "compile_by_trace": True,
}


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_build_fstring_case_1():
    """
    Feature: String operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn(s1: str):
        return f'{s1}-xxx'

    context.set_context(mode=context.PYNATIVE_MODE)
    s1 = fn('hello')

    fn = jit(fn, mode="PIJit", jit_config=cfg)
    s2 = fn('hello')

    assert s1 == s2
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_build_fstring_case_2():
    """
    Feature: String operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn(s1: str, s2: str):
        return f'{s1}{s2}-xxx'

    context.set_context(mode=context.PYNATIVE_MODE)
    s1 = fn('hello', ' world')

    fn = jit(fn, mode="PIJit", jit_config=cfg)
    s2 = fn('hello', ' world')

    assert s1 == s2
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_string_add_string():
    """
    Feature: String operation.
    Description: Test one stage basic operation.
    Expectation: No exception.
    """
    def fn(s1: str, s2: str):
        return s1 + s2

    context.set_context(mode=context.PYNATIVE_MODE)
    s1 = fn('hello', ' world')

    fn = jit(fn, mode="PIJit", jit_config=cfg)
    s2 = fn('hello', ' world')

    assert s1 == s2
    assert_executed_by_graph_mode(fn)
