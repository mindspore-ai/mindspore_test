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
"""test graph list comprehension"""
import numpy as np

from mindspore import Tensor, jit, context

context.set_context(mode=context.GRAPH_MODE)


def test_list_comprehension_with_local_inputs():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with local input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [i for i in range(3)]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([0, 1, 2]))


def test_nested_list_comprehension_with_local_inputs():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with local input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [i + j + 1 for i in range(3) for j in range(2)]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([1, 2, 2, 3, 3, 4]))


def test_list_comprehension_with_local_inputs_2():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with local input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [i + 1 for i in range(3)]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([1, 2, 3]))


def test_list_comprehension_with_local_inputs_and_condition():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with local input.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [i + 1 for i in range(5) if i%2 == 0]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([1, 3, 5]))


def test_nested_list_comprehension_with_local_inputs_and_condition():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with local input and condition.
    Expectation: No exception.
    """

    @jit
    def foo():
        x = [i + j + 1 for i in range(5) if i%2 == 0 for j in range(4) if (i+j)%2 == 1]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([2, 4, 4, 6, 6, 8]))


def test_list_comprehension_with_pre_block_local_input():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with local input from previous block.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        x = [a for i in range(3)]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([10, 10, 10]))


def test_nested_list_comprehension_with_pre_block_local_input():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with local input from previous block.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        x = [a for i in range(3) for j in range(2)]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([10, 10, 10, 10, 10, 10]))


def test_list_comprehension_with_pre_block_local_input_2():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with local input from previous block.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        x = [a + i for i in range(3)]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([10, 11, 12]))


def test_nested_list_comprehension_with_pre_block_local_input_2():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with local input from previous block.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        x = [a + i - j for i in range(3) for j in range(2)]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([10, 9, 11, 10, 12, 11]))


def test_list_comprehension_with_pre_block_local_input_and_condition():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with local input from previous block.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        x = [a + i for i in range(3) if a > 5]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([10, 11, 12]))


def test_nested_list_comprehension_with_pre_block_local_input_and_condition():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with local input from previous block.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        x = [a + i - j for i in range(3) if a > 5 for j in range(4) if j%2 == 1]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([9, 7, 10, 8, 11, 9]))


def test_list_comprehension_with_pre_block_local_input_and_condition_2():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with local input from previous block.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        x = [a + i for i in range(5) if a + i < 13]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([10, 11, 12]))


def test_nested_list_comprehension_with_pre_block_local_input_and_condition_2():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with local input from previous block.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        x = [a + i - j for i in range(5) if a + i < 13 for j in range(3) if a + j < 12]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([10, 9, 11, 10, 12, 11]))


def test_list_comprehension_with_numpy_input():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with numpy input.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = np.array([1, 2, 3])
        x = [a for i in range(3)]
        return Tensor(x[0]), Tensor(x[1]), Tensor(x[2])

    res = foo()
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([1, 2, 3]))


def test_nested_list_comprehension_with_numpy_input():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with numpy input.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = np.array([1, 2, 3])
        x = [a for i in range(2) for j in range(2)]
        return Tensor(x[0]), Tensor(x[1]), Tensor(x[2]), Tensor(x[3])

    res = foo()
    assert len(res) == 4
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[3].asnumpy() == np.array([1, 2, 3]))


def test_list_comprehension_with_numpy_input_2():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with numpy input.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = np.array([1, 2, 3])
        x = [a + i for i in range(3)]
        return Tensor(x[0]), Tensor(x[1]), Tensor(x[2])

    res = foo()
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([2, 3, 4]))
    assert np.all(res[2].asnumpy() == np.array([3, 4, 5]))


def test_nested_list_comprehension_with_numpy_input_2():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with numpy input.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = np.array([1, 2, 3])
        x = [a + i + j for i in range(2) for j in range(2)]
        return Tensor(x[0]), Tensor(x[1]), Tensor(x[2]), Tensor(x[3])

    res = foo()
    assert len(res) == 4
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([2, 3, 4]))
    assert np.all(res[2].asnumpy() == np.array([2, 3, 4]))
    assert np.all(res[3].asnumpy() == np.array([3, 4, 5]))


def test_list_comprehension_with_numpy_input_and_condition():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with numpy input and condition.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = np.array([1, 2, 3])
        x = [a for i in range(5) if i%2 == 0]
        return Tensor(x[0]), Tensor(x[1]), Tensor(x[2])

    res = foo()
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([1, 2, 3]))


def test_nested_list_comprehension_with_numpy_input_and_condition():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with numpy input and condition.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = np.array([1, 2, 3])
        x = [a for i in range(5) if i%2 == 0 for j in range(5) if j < 1]
        return Tensor(x[0]), Tensor(x[1]), Tensor(x[2])

    res = foo()
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([1, 2, 3]))


def test_list_comprehension_with_numpy_input_and_condition_2():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with numpy input and condition.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = np.array([1, 2, 3])
        x = [a + i for i in range(5) if np.sum(a + i) > 10]
        return Tensor(x[0]), Tensor(x[1]), Tensor(x[2])

    res = foo()
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([3, 4, 5]))
    assert np.all(res[1].asnumpy() == np.array([4, 5, 6]))
    assert np.all(res[2].asnumpy() == np.array([5, 6, 7]))


def test_nested_list_comprehension_with_numpy_input_and_condition_2():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with numpy input and condition.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = np.array([1, 2, 3])
        x = [a + i + j for i in range(4) if np.sum(a + i) > 10 for j in range(4) if np.sum(a + j) > 10]
        return Tensor(x[0]), Tensor(x[1]), Tensor(x[2]), Tensor(x[3])

    res = foo()
    assert len(res) == 4
    assert np.all(res[0].asnumpy() == np.array([5, 6, 7]))
    assert np.all(res[1].asnumpy() == np.array([6, 7, 8]))
    assert np.all(res[2].asnumpy() == np.array([6, 7, 8]))
    assert np.all(res[3].asnumpy() == np.array([7, 8, 9]))


def test_list_comprehension_with_numpy_input_and_condition_3():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with numpy input and condition.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = np.array([1, 2, 3])
        x = [a + i for i in range(5) if np.sum(a + i) > 20]
        return x

    res = foo()
    assert not res


def test_nested_list_comprehension_with_numpy_input_and_condition_3():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with numpy input and condition.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = np.array([1, 2, 3])
        x = [a + i for i in range(5) if np.sum(a + i) > 20 for j in range(5) if np.sum(a + j) > 20]
        return x

    res = foo()
    assert not res


def test_list_comprehension_with_iter_list():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with list as iteration object.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        m = [1, 2, 3, 4, 5]
        x = [a + i for i in m if (a + i)%2 == 0]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([12, 14]))


def test_nested_list_comprehension_with_iter_list():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with list as iteration object.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        m = [1, 2, 3, 4, 5]
        n = [2, 3, 4, 5, 6]
        x = [a + i + j for i in m if (a + i)%2 == 0 for j in n if (i + j)%2 == 1]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([15, 17, 17, 19]))


def test_list_comprehension_with_iter_list_2():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with list as iteration object.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        m = np.array([1, 2, 3, 4, 5])
        x = [a + i for i in m if (a + i)%2 == 0]
        return Tensor(x)

    res = foo()
    assert np.all(res.asnumpy() == np.array([12, 14]))


def test_list_comprehension_with_iter_list_3():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with list as iteration object.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        m = [Tensor([1]), Tensor([2]), Tensor([3])]
        x = [a + i for i in m]
        return x[0], x[1], x[2]

    res = foo()
    assert len(res) == 3
    assert res[0] == 11
    assert res[1] == 12
    assert res[2] == 13


def test_nested_list_comprehension_with_iter_list_3():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with list as iteration object.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = 10
        m = [Tensor([1]), Tensor([2])]
        n = [Tensor([2]), Tensor([3])]
        x = [a + i + j for i in m for j in n]
        return x[0], x[1], x[2], x[3]

    res = foo()
    assert len(res) == 4
    assert res[0] == 13
    assert res[1] == 14
    assert res[2] == 14
    assert res[3] == 15


def test_list_comprehension_with_string():
    """
    Feature: Graph isinstance.
    Description: Graph list comprehension syntax with list as iteration object.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = "abcdef"
        return tuple([i for i in a])

    res = foo()
    assert res == ('a', 'b', 'c', 'd', 'e', 'f')


def test_nested_list_comprehension_with_string():
    """
    Feature: Graph isinstance.
    Description: Nested graph list comprehension syntax with a string as iteration object.
    Expectation: No exception.
    """

    @jit
    def foo():
        a = "abc"
        b = "bc"
        return tuple([i+j for i in a for j in b])

    res = foo()
    assert res == ('ab', 'ac', 'bb', 'bc', 'cb', 'cc')
