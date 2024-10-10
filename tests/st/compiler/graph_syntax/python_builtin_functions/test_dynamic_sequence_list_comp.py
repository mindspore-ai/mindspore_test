# Copyright 2023 Huawei Technologies Co., Ltd
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
"""test ListComprehension for dynamic sequence in graph mode"""
import pytest
from mindspore.common import mutable
from mindspore import jit, nn, Tensor
from mindspore import context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@pytest.mark.skip(reason='temporarily skip this case to pass ci')
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dynamic_sequence_list_comp_1():
    """
    Feature: ListComprehension with dynamic length sequence.
    Description: support dynamic length sequence as the input of ListComprehension
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable((100, 200, 300, 400), True)
        out = [i + 1 for i in x]
        return out

    res = foo()
    assert res == [101, 201, 301, 401]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dynamic_sequence_list_comp_2():
    """
    Feature: ListComprehension with dynamic length sequence.
    Description: support dynamic length sequence as the input of ListComprehension
    Expectation: No exception.
    """
    @jit
    def foo(x):
        out = [i + 1 for i in x]
        return out

    x = mutable((100, 200, 300, 400), True)
    res = foo(x)
    assert res == [101, 201, 301, 401]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dynamic_sequence_list_comp_3():
    """
    Feature: ListComprehension with dynamic length sequence.
    Description: support dynamic length sequence as the input of ListComprehension
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self, x):
            x = [i for i in range(len(x))]
            return x

    net = InnerClass()
    res = net(mutable([Tensor(1), Tensor(2), Tensor(3), Tensor(4)], True))
    assert res == [0, 1, 2, 3]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dynamic_sequence_list_comp_with_condition_2():
    """
    Feature: ListComprehension with dynamic length sequence and condition.
    Description: support dynamic length sequence as the input of ListComprehension
    Expectation: No exception.
    """
    @jit
    def foo(x):
        out = [i + 1 for i in x if i > 200]
        return out

    x = mutable((100, 200, 300, 400), True)
    res = foo(x)
    assert res == [301, 401]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dynamic_sequence_list_comp_with_condition_3():
    """
    Feature: ListComprehension with dynamic length sequence and condition.
    Description: support dynamic length sequence as the input of ListComprehension
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self, x):
            x = [i for i in range(len(x)) if i > 0]
            return x

    net = InnerClass()
    res = net(mutable([Tensor(1), Tensor(2), Tensor(3), Tensor(4)], True))
    assert res == [1, 2, 3]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_dynamic_sequence_list_comp_with_condition_4():
    """
    Feature: Single ListComprehension with dynamic length sequence and condition. With expected output: empty list.
    Description: support dynamic length sequence as the input of ListComprehension
    Expectation: No exception.
    """
    @jit()
    def foo(x):
        out = [i + 1 for i in x if i > 400]
        return out

    x = mutable((100, 200, 300), True)
    res = foo(x)
    assert res == []


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_nested_dynamic_sequence_list_comp_1():
    """
    Feature: Nested ListComprehension with dynamic length sequence.
    Description: support dynamic length sequence as the input of NestedListComprehension
    Expectation: No exception.
    """
    @jit()
    def foo(x):
        out = [i + j + 1 for i in x for j in x]
        return out

    x = mutable((100., 200.), True)
    res = foo(x)
    assert res == [201., 301., 301., 401.]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_nested_dynamic_sequence_list_comp_2():
    """
    Feature:  Nested ListComprehension with dynamic length sequence.
    Description: support dynamic length sequence as the input of ListComprehension
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self, x):
            x = [i + j for i in range(len(x)) for j in range(len(x))]
            return x

    net = InnerClass()
    res = net(mutable([Tensor(1), Tensor(2)], True))
    assert res == [0, 1, 1, 2]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_nested_dynamic_sequence_list_comp_with_condition_2():
    """
    Feature: Nested ListComprehension with dynamic length sequence and condition.
    Description: support dynamic length sequence as the input of ListComprehension
    Expectation: No exception.
    """
    class InnerClass(nn.Cell):
        def construct(self, x):
            x = [i + j for i in range(len(x)) for j in range(len(x)) if i > 0]
            return x

    net = InnerClass()
    res = net(mutable([Tensor(1), Tensor(2)], True))
    assert res == [1, 2]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_nested_dynamic_sequence_list_comp_with_condition_3():
    """
    Feature: Nested ListComprehension with dynamic length sequence and condition. With expected output: empty list.
    Description: support dynamic length sequence as the input of NestedListComprehension
    Expectation: No exception.
    """
    @jit()
    def foo(x):
        out = [i + j + 1 for i in x for j in x if i > 200 if j < 0]
        return out

    x = mutable((100, 200, 300), True)
    res = foo(x)
    assert res == []
