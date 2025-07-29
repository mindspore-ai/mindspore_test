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
"""Test python standard library: operator"""
import operator
import pytest

import mindspore
from mindspore import jit, Tensor
from mindspore.common._pijit_context import Unsupported

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_equal, assert_executed_by_graph_mode


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_operator_index_of_const_int_v1():
    """
    Feature: operator.index().
    Description: operator.index() of const int.
    Expectation: No graph break.
    """

    def fn(x: Tensor, values: tuple, idx: int):
        idx = operator.index(idx)
        return x + values[idx]

    x = mindspore.tensor([1, 2])
    values = (1, 2, 3)
    idx = 1
    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)

    o1 = fn(x, values, idx)
    o2 = compiled_fn(x, values, idx)
    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn, call_count=1)

    idx = 2
    o3 = fn(x, values, idx)
    o4 = compiled_fn(x, values, idx)
    assert_equal(o3, o4)
    assert_executed_by_graph_mode(compiled_fn, call_count=1)  # need recompile, so call_count=1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_operator_index_of_const_int_v2():
    """
    Feature: operator.index().
    Description: operator.index() of const int.
    Expectation: No graph break.
    """

    def fn(x: Tensor, idx: int):
        idx = operator.index(idx)
        return x + x[idx]

    x = mindspore.tensor([1, 2, 3])
    idx = 1
    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)

    o1 = fn(x, idx)
    o2 = compiled_fn(x, idx)
    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn, call_count=1)

    idx = 2
    o3 = fn(x, idx)
    o4 = compiled_fn(x, idx)
    assert_equal(o3, o4)
    assert_executed_by_graph_mode(compiled_fn, call_count=1)  # need recompile, so call_count=1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_operator_index_of_const_int_v3():
    """
    Feature: operator.index().
    Description: operator.index() of const int.
    Expectation: No graph break.
    """

    def fn(x: Tensor, idx: int):
        return f2(x, idx + 1)

    def f2(x: Tensor, idx: int):
        idx = operator.index(idx)
        return x + x[idx]

    x = mindspore.tensor([1, 2, 3, 4])
    idx = 1
    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)

    o1 = fn(x, idx)
    o2 = compiled_fn(x, idx)
    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn, call_count=1)

    idx = 2
    o3 = fn(x, idx)
    o4 = compiled_fn(x, idx)
    assert_equal(o3, o4)
    assert_executed_by_graph_mode(compiled_fn, call_count=1)  # need recompile, so call_count=1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_operator_index_of_mutable_int_v1():
    """
    Feature: operator.index().
    Description: operator.index() of mutable int.
    Expectation: No graph break.
    """

    def fn(x: Tensor, values: tuple, idx: int):
        idx = operator.index(idx)
        return x + values[idx]

    x = mindspore.tensor([1, 2])
    values = (1, 2, 3)
    idx = mindspore.mutable(1)
    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)

    o1 = fn(x, values, idx)
    o2 = compiled_fn(x, values, idx)
    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn, call_count=1)

    idx = mindspore.mutable(2)
    o3 = fn(x, values, idx)
    o4 = compiled_fn(x, values, idx)
    assert_equal(o3, o4)
    assert_executed_by_graph_mode(compiled_fn, call_count=2)  # no need recompile, so call_count=2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_operator_index_of_mutable_int_v2():
    """
    Feature: operator.index().
    Description: operator.index() of mutable int.
    Expectation: No graph break.
    """

    def fn(x: Tensor, idx: int):
        idx = operator.index(idx)
        return x + x[idx]

    x = mindspore.tensor([1, 2, 3])
    idx = mindspore.mutable(1)
    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)

    o1 = fn(x, idx)
    o2 = compiled_fn(x, idx)
    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn, call_count=1)

    idx = mindspore.mutable(2)
    o3 = fn(x, idx)
    o4 = compiled_fn(x, idx)
    assert_equal(o3, o4)
    assert_executed_by_graph_mode(compiled_fn, call_count=2)  # no need recompile, so call_count=2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_operator_index_of_mutable_int_v3():
    """
    Feature: operator.index().
    Description: operator.index() of mutable int.
    Expectation: No graph break.
    """

    def fn(x: Tensor, idx: int):
        return f2(x, idx + 1)

    def f2(x: Tensor, idx: int):
        idx = operator.index(idx)
        return x + x[idx]

    x = mindspore.tensor([1, 2, 3, 4])
    idx = mindspore.mutable(1)
    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)

    o1 = fn(x, idx)
    o2 = compiled_fn(x, idx)
    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn, call_count=1)

    idx = mindspore.mutable(2)
    o3 = fn(x, idx)
    o4 = compiled_fn(x, idx)
    assert_equal(o3, o4)
    assert_executed_by_graph_mode(compiled_fn, call_count=2)  # no need recompile, so call_count=2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_operator_index_of_unsupported_param():
    """
    Feature: operator.index().
    Description: operator.index() of Tensor object, which is unsupported.
    Expectation: Graph break.
    """

    @jit(capture_mode='bytecode', fullgraph=True)
    def fn(x: Tensor, idx: Tensor):
        idx = operator.index(idx)
        return x + x[idx]

    x = mindspore.tensor([1, 2, 3])
    idx = mindspore.tensor(1)
    with pytest.raises(Unsupported) as err_info:
        o = fn(x, idx)
        s = str(err_info.value)
        assert 'idx = operator.index(idx)' in s
        assert 'Reason: Unsupported function' in s
