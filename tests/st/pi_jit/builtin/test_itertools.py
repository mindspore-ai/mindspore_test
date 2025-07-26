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
"""Test python standard library: itertools"""
import collections
from collections.abc import Iterable
import itertools
import pytest
from typing import List

import mindspore
from mindspore import jit, Tensor

from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import assert_equal, assert_executed_by_graph_mode


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_itertools_chain_v1():
    """
    Feature: itertools.chain.
    Description: for-loop of itertools.chain().
    Expectation: No graph break.
    """

    def fn(x: Tensor, int_values: tuple, tensor_values: list):
        for i in itertools.chain(int_values, tensor_values):
            x = x + i
        return x

    x = mindspore.tensor([1, 2])
    int_values = (1, 2)
    tensor_values = [mindspore.tensor([1, 1]), mindspore.tensor([2, 2])]

    o1 = fn(x, int_values, tensor_values)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(x, int_values, tensor_values)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_itertools_chain_v2():
    """
    Feature: itertools.chain.
    Description: for-loop of itertools.chain().
    Expectation: No graph break.
    """

    def fn(x: Tensor, values: list):
        for i in itertools.chain(*values):
            x = x + i
        return x

    x = mindspore.tensor([1, 2])
    values = [(1, 2), (mindspore.tensor([1, 1]), mindspore.tensor([1, 2])), (2.0, mindspore.tensor([3.0]))]

    o1 = fn(x, values)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(x, values)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_itertools_chain_v3():
    """
    Feature: itertools.chain.
    Description: for-loop of itertools.chain().
    Expectation: No graph break.
    """

    def fn(x: Tensor, *values: Iterable):
        for i in itertools.chain(*values):
            x = x + i
        return x

    x = mindspore.tensor([1, 2])
    int_values = (1, 2)
    tensor_values = [mindspore.tensor([1, 1]), mindspore.tensor([2, 2])]

    o1 = fn(x, int_values, tensor_values)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(x, int_values, tensor_values)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_itertools_chain_from_iterable_v1():
    """
    Feature: itertools.chain.from_iterable.
    Description: for-loop of itertools.chain.from_iterable().
    Expectation: No graph break.
    """

    def fn(x: Tensor, values: list):
        for i in itertools.chain.from_iterable(values):
            x = x + i
        return x

    x = mindspore.tensor([1, 2])
    values = [(1, 2), (mindspore.tensor([1, 1]), mindspore.tensor([1, 2])), (2.0, mindspore.tensor([3.0]))]

    o1 = fn(x, values)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(x, values)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_itertools_chain_from_iterable_v2():
    """
    Feature: itertools.chain.from_iterable.
    Description: for-loop of itertools.chain.from_iterable().
    Expectation: No graph break.
    """

    def _as_iterable(obj) -> collections.abc.Iterable:
        return obj if isinstance(obj, list) else (obj,)

    def _ensure_all_tensors_same_dtype(*tensors) -> None:
        last_dtype = None
        tensors = [_as_iterable(group) for group in tensors]
        for tensor in itertools.chain.from_iterable(tensors):
            tensor_dtype = tensor.dtype
            if last_dtype is None:
                last_dtype = tensor_dtype
            else:
                if last_dtype != tensor_dtype:
                    raise TypeError("Invalid usage of tensors with different dtypes")

    def fn(input_tensors: List[Tensor], output_tensor: Tensor):
        _ensure_all_tensors_same_dtype(input_tensors, output_tensor)
        for tensor in itertools.chain.from_iterable([input_tensors, [output_tensor]]):
            output_tensor = output_tensor + tensor
        return output_tensor

    input_tensors = [mindspore.tensor([1, 2]), mindspore.tensor([3, 4])]
    output_tensor = mindspore.tensor([0, 0])

    o1 = fn(input_tensors, output_tensor)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(input_tensors, output_tensor)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_itertools_chain_from_iterable_v3():
    """
    Feature: itertools.chain.from_iterable.
    Description: for-loop of itertools.chain.from_iterable().
    Expectation: No graph break.
    """

    def fn(x: Tensor, *values: Iterable):
        for i in itertools.chain.from_iterable(values):
            x = x + i
        return x

    x = mindspore.tensor([1, 2])
    int_values = (1, 2)
    tensor_values = [mindspore.tensor([1, 1]), mindspore.tensor([2, 2])]

    o1 = fn(x, int_values, tensor_values)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(x, int_values, tensor_values)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_itertools_chain_from_iterable_and_zip_v1():
    """
    Feature: itertools.chain.from_iterable.
    Description: for-loop of itertools.chain.from_iterable() + zip().
    Expectation: No graph break.
    """

    def fn(x: Tensor, values_a: list, values_b: tuple):
        for i in itertools.chain.from_iterable(zip(values_a, values_b)):
            x = x + i
        return x

    x = mindspore.tensor([1, 2, 3])
    values_a = [1, 2]
    values_b = (3, 4)

    o1 = fn(x, values_a, values_b)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(x, values_a, values_b)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_itertools_chain_from_iterable_and_zip_v2():
    """
    Feature: itertools.chain.from_iterable.
    Description: for-loop of itertools.chain.from_iterable() + zip().
    Expectation: No graph break.
    """

    def fn(x: Tensor, values_a: list, values_b: tuple):
        for i in itertools.chain.from_iterable(zip(values_a, values_b)):
            x = x + i
        return x

    x = mindspore.tensor([1, 2])
    values_a = [mindspore.tensor([1, 1]), mindspore.tensor([2, 2])]
    values_b = (mindspore.tensor([3, 3]), mindspore.tensor([4, 4]))

    o1 = fn(x, values_a, values_b)

    compiled_fn = jit(fn, capture_mode='bytecode', fullgraph=True)
    o2 = compiled_fn(x, values_a, values_b)

    assert_equal(o1, o2)
    assert_executed_by_graph_mode(compiled_fn)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_itertools_chain_from_iterable_with_illegal_input():
    """
    Feature: itertools.chain.from_iterable.
    Description: The input of itertools.chain.from_iterable() is illegal.
    Expectation: todo.
    """

    @jit(capture_mode='bytecode', fullgraph=True)
    def fn(x: Tensor, values: list):
        for i in itertools.chain.from_iterable(values):
            x = x + i
        return x

    x = mindspore.tensor([1, 2, 3])
    values = [1, 2, 3, 4]  # user code bug! should be nested iterable.

    with pytest.raises(TypeError) as err_info:
        o = fn(x, values)
        assert "TypeError: 'int' object is not iterable" in str(err_info.value)
