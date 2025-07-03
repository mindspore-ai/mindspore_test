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
"""Test @jit(fullgraph=True)"""
import pytest

from mindspore import Tensor, ops, context, jit
from mindspore.common._pijit_context import Unsupported

from tests.st.pi_jit.share.utils import pi_jit_with_config
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_1():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True, and there's a graph-break in tested function.
    Expectation: Throw exception.
    """

    @jit(capture_mode='bytecode', fullgraph=True)
    def fn(x: Tensor):
        x = x + 1
        print('Graph break!', flush=True)  # graph break
        return x * 2

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ops.randn(2, 2)
    with pytest.raises(Unsupported) as info:
        o = fn(x)
    assert "print('Graph break!', flush=True)" in str(info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_2():
    """
    Feature: @jit(fullgraph=True).
    Description: 1.fullgraph=True; 2.loop_unrolling=False; 3.there's a for-loop in tested function.
    Expectation: Throw exception.
    """
    jit_cfg = {'loop_unrolling': False}

    @pi_jit_with_config(jit_config=jit_cfg, fullgraph=True)
    def fn(x: Tensor):
        for i in range(3):  # graph break!
            x = x + 1
        return x

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ops.randn(2, 2)
    with pytest.raises(Unsupported) as info:
        o = fn(x)
    assert 'for i in range(3):' in str(info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_3():
    """
    Feature: @jit(fullgraph=True).
    Description: 1.fullgraph=True; 2.there are unsupported bytecodes(LOAD_BUILD_CLASS, LOAD_CLASSDEREF).
    Expectation: Throw exception.
    """

    @jit(capture_mode='bytecode', fullgraph=True)
    def fn():
        x = 42

        class Inner:  # LOAD_BUILD_CLASS, unsupported bytecode.
            y = x  # x is freevar

        return Inner

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(Unsupported) as info:
        o = fn()
    err_msg = str(info.value)
    assert 'class Inner:' in err_msg
    assert 'Hint: ByteCode LOAD_BUILD_CLASS is not supported' in err_msg
    assert 'Hint: See https://docs.python.org/3/library/dis.html for bytecode semantics' in err_msg


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_4():
    """
    Feature: @jit(fullgraph=True).
    Description: fullgraph=True, and there's a graph-break in inner function.
    Expectation: Throw exception.
    """

    def inner():
        print('Graph break in inner function!', flush=True)  # graph break

    @jit(capture_mode='bytecode', fullgraph=True)
    def fn(x: Tensor):
        x = x + 1
        inner()
        return x * 2

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ops.randn(2, 2)
    with pytest.raises(Unsupported) as info:
        o = fn(x)
    err_msg = str(info.value)
    assert "inner()" in err_msg
    assert "print('Graph break in inner function!', flush=True)" in err_msg
    assert err_msg.find("inner()") < err_msg.find("print('Graph break in inner function!', flush=True)")


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_5():
    """
    Feature: @jit(fullgraph=True).
    Description: 1.fullgraph=True; 2.unsupported iterable type.
    Expectation: Throw exception.
    """

    @jit(capture_mode='bytecode', fullgraph=True)
    def fn(seq: set):
        ret = 0
        for i in seq:  # graph break!
            ret = ret + i
        return ret

    context.set_context(mode=context.PYNATIVE_MODE)
    with pytest.raises(Unsupported) as info:
        s = {1, 2, 3}
        o = fn(s)
    err_msg = str(info.value)
    assert 'for i in seq:' in err_msg
    assert 'Hint: Unsupported iterable type: set' in err_msg


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_fullgraph_True_and_compile_with_try_True():
    """
    Feature: @jit(fullgraph=True).
    Description: 1.fullgraph=True; 2.compile_with_try=True.
    Expectation: Throw exception.
    """

    @pi_jit_with_config(jit_config={'compile_with_try': True}, fullgraph=True)
    def fn(x: Tensor):
        x = x + 1
        print('Graph break!', flush=True)  # graph break
        return x * 2

    context.set_context(mode=context.PYNATIVE_MODE)
    x = ops.randn(2, 2)
    with pytest.raises(Unsupported) as info:
        o = fn(x)
    assert "print('Graph break!', flush=True)" in str(info.value)
