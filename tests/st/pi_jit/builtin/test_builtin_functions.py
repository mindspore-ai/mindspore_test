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

from mindspore import Tensor, ops, jit, context

from tests.st.pi_jit.share.utils import match_array, assert_executed_by_graph_mode, pi_jit_with_config
from tests.mark_utils import arg_mark


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

    fn = jit(fn, capture_mode='bytecode')
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

    @jit(capture_mode='bytecode')
    def fn(x: Tensor):
        return len(x) + 1

    context.set_context(mode=context.PYNATIVE_MODE)
    x = Tensor([1, 2, 3, 4])
    o = fn(x)

    assert o == 5


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

    fn = jit(fn, capture_mode='bytecode')
    o2 = fn(x)

    match_array(o1, o2)
    assert_executed_by_graph_mode(fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_builtin_function_type_v1():
    """
    Feature: python builtin type().
    Description: Test one stage basic operation.
    Expectation: No graph breaks.
    """

    def view(x: Tensor, *shape):
        # when x triggers dynamic shape, the shape argument may become a variable (contains kValueAny).
        if type(shape) is tuple:
            return ops.reshape(x, shape)
        else:
            return ops.flatten(x)

    def fn(x: Tensor, n: int, dim: int):
        B = x.shape[0]
        T = x.shape[1]  # may trigger dynamic shape
        return view(x, B, T, n, dim)

    compiled_fn = pi_jit_with_config(fn, jit_config={'_symbolic': 1}, fullgraph=True)

    # Currently, the 7th tensor shape change triggers dynamic shape compilation.
    for i in range(1, 10):
        x = ops.randn(1, i, 4)
        o1 = fn(x, 2, 2)
        o2 = compiled_fn(x, 2, 2)
        match_array(o1, o2)
        assert_executed_by_graph_mode(compiled_fn)
