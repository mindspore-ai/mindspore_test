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
"""Test return list type object from graph"""
import pytest
import numpy as np

import mindspore.common.dtype as mstype
from mindspore import ops
from mindspore.common import mutable
from mindspore import Tensor, jit, context
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    @jit
    def foo():
        return [1, 2, 3, 4]

    res = foo()
    assert res == [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_2():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    @jit
    def foo():
        return ["a", "b", "c", "d"]

    res = foo()
    assert res == ["a", "b", "c", "d"]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_3():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    @jit
    def foo():
        return [True, False, False, True]

    res = foo()
    assert res == [True, False, False, True]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_4():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    @jit
    def foo():
        return [Tensor([1]), Tensor([1, 2, 3]), Tensor([2, 3])]

    res = foo()
    assert len(res) == 3
    assert np.all(res[0].asnumpy() == np.array([1]))
    assert np.all(res[1].asnumpy() == np.array([1, 2, 3]))
    assert np.all(res[2].asnumpy() == np.array([2, 3]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_5():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    @jit
    def foo():
        return [None, None, None]

    res = foo()
    assert res == [None, None, None]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_6():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    @jit
    def foo():
        return [np.array([1, 2, 3]), np.array([4, 5, 6]), 1]

    res = foo()
    assert isinstance(res, list)
    assert len(res) == 3
    assert np.all(res[0] == np.array([1, 2, 3]))
    assert np.all(res[1] == np.array([4, 5, 6]))
    assert res[2] == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_constant_list_7():
    """
    Feature: Return list in graph
    Description: Support return constant list.
    Expectation: No exception.
    """
    @jit
    def foo():
        return [1, "a", True, None, Tensor([2])]

    res = foo()
    assert res == [1, "a", True, None, Tensor([2])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_node():
    """
    Feature: Return list in graph
    Description: Support return make list node.
    Expectation: No exception.
    """
    context.set_context(jit_level='O0')
    @jit
    def foo(x):
        return [x, x+1, x+2, 1]

    res = foo(mutable(1))
    assert res == [1, 2, 3, 1]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_node_2():
    """
    Feature: Return list in graph
    Description: Support return make list node.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return [x, x+1, x+2, Tensor([4])]

    res = foo(Tensor([1]))
    assert res == [Tensor([1]), Tensor([2]), Tensor([3]), Tensor([4])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_node_3():
    """
    Feature: Return list in graph
    Description: Support return make list node.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return [x, mutable(1), "a"]

    res = foo(Tensor([1]))
    assert res == [Tensor([1]), 1, "a"]


@pytest.mark.skip('backend not support different type in value tuple')
@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_node_4():
    """
    Feature: Return list in graph
    Description: Support return make list node.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        x1 = list(x)
        x2 = {"a": Tensor(5)}
        x3 = (0, 1.0)
        return [x1, x2, x3]

    res = foo(Tensor([1, 2, 3]))
    assert isinstance(res, list)
    assert len(res) == 3
    assert res[0] == [Tensor([1]), Tensor([2]), Tensor([3])]
    assert res[1] == {"a": Tensor(5)}
    assert res[2] == (0, 1.0)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_with_nest():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        return [[1, 2, 3], [4, 5, 6]]

    res = foo()
    assert res == [[1, 2, 3], [4, 5, 6]]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_with_nest_2():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        return [([1, 1], [2, 2], (3, [4, 4])), [4, 5, 6]]

    res = foo()
    assert res == [([1, 1], [2, 2], (3, [4, 4])), [4, 5, 6]]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_with_nest_3():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """
    @jit
    def foo():
        return (([1, 1], [2, 2], (3, [4, 4])), [4, 5, 6])

    res = foo()
    assert res == (([1, 1], [2, 2], (3, [4, 4])), [4, 5, 6])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_with_nest():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return [[x, x], (x+1, x+2)]

    res = foo(Tensor([0]))
    assert res == [[Tensor([0]), Tensor([0])], (Tensor([1]), Tensor([2]))]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_make_list_with_nest_2():
    """
    Feature: Return list in graph
    Description: Support return make list in nest scene.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        return [x, ([x, 1],)], (x+1, x+2)

    res = foo(Tensor([0]))
    assert res == ([Tensor([0]), ([Tensor([0]), 1],)], (Tensor([1]), Tensor([2])))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_buildin_list_func():
    """
    Feature: Return list in graph
    Description: Support return result of list() function.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        return list((1, "2", None, Tensor([1])))

    res = foo()
    assert res == [1, "2", None, Tensor([1])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_buildin_list_func_2():
    """
    Feature: Return list in graph
    Description: Support return result of list() function.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x):
        return list(x)

    res = foo(Tensor([1, 2, 3]))
    assert res == [Tensor([1]), Tensor([2]), Tensor([3])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_dynamic_length_list():
    """
    Feature: Return list in graph
    Description: Support return dynamic length list.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = mutable([1, 2, 3], True)
        return x

    res = foo()
    assert res == [1, 2, 3]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_dynamic_length_list_2():
    """
    Feature: Return list in graph
    Description: Support return dynamic length list.
    Expectation: No exception.
    """
    @jit
    def foo(m):
        x = mutable([m, m+1], True)
        return x

    res = foo(Tensor([0]))
    assert res == [Tensor([0]), Tensor([1])]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_from_third_party():
    """
    Feature: Return list in graph
    Description: Support return list from third party.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo():
        m = np.array([1, 2, 3, 4])
        x = m.tolist()
        return x

    res = foo()
    assert res == [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_from_third_party_2():
    """
    Feature: Return list in graph
    Description: Support return list from third party.
    Expectation: No exception.
    """
    @jit
    def foo(m):
        x = m.asnumpy().tolist()
        return x

    res = foo(Tensor([1, 2, 3, 4]))
    assert res == [1, 2, 3, 4]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_from_third_party_3():
    """
    Feature: Return list in graph
    Description: Support return list from third party.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = np.arange(0, 10, 2)
        return list(x)

    res = foo()
    assert res == [0, 2, 4, 6, 8]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_from_dict_attribute():
    """
    Feature: Return list in graph
    Description: Support return list from dict keys and values.
    Expectation: No exception.
    """
    @jit(backend="ms_backend")
    def foo(x, y):
        m = {"1": x, "2": y}
        return list(m.keys()), list(m.values())

    res = foo(Tensor([1]), mutable(2))
    assert len(res) == 2
    assert res[0] == ["1", "2"]
    assert res[1] == [Tensor([1]), 2]


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_return_list_from_dict_attribute_2():
    """
    Feature: Return list in graph
    Description: Support return list from dict keys and values.
    Expectation: No exception.
    """
    @jit
    def foo(x, y):
        m = {"1": x, "2": y}
        return m, list(m.keys()), list(m.values())

    res = foo(Tensor([1]), mutable(2))
    assert len(res) == 3
    assert res[1] == ["1", "2"]
    assert res[2] == [Tensor([1]), 2]



@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_grad_for_return_list_graph():
    """
    Feature: Return list in graph
    Description: Support calculate gradient for graph with list return.
    Expectation: No exception.
    """
    @jit
    def foo(x):
        y = ops.ReLU()(x)
        return [y,]

    x = Tensor([[0.8, 0.6, 0.2], [1.8, 1.3, 1.1]], dtype=mstype.float32)
    res = ops.grad(foo)(x)  # pylint: disable=not-callable
    assert np.allclose(res.asnumpy(), np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]).astype(np.float32))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_grad_for_graph_with_list_input():
    """
    Feature: Return list in graph
    Description: Support calculate gradient for graph with list return.
    Expectation: No exception.
    """
    @jit
    def foo(t):
        x = t[0]
        y = t[1]
        out = ops.MatMul()(x, y)
        return out

    t = mutable([Tensor([[0.5, 0.6, 0.4], [1.2, 1.3, 1.1]], dtype=mstype.float32),
                 Tensor([[0.01, 0.3, 1.1], [0.1, 0.2, 1.3], [2.1, 1.2, 3.3]], dtype=mstype.float32)])
    output = ops.grad(foo)(t)  # pylint: disable=not-callable
    assert isinstance(output, list)
    expect = [np.array([[1.4100001, 1.5999999, 6.6],
                        [1.4100001, 1.5999999, 6.6]]).astype(np.float32),
              np.array([[1.7, 1.7, 1.7],
                        [1.9, 1.9, 1.9],
                        [1.5, 1.5, 1.5]]).astype(np.float32)]
    assert np.allclose(output[0].asnumpy(), expect[0])
    assert np.allclose(output[1].asnumpy(), expect[1])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level2', card_mark='onecard',
          essential_mark='unessential')
def test_fallback_sorted_return_list():
    """
    Feature: Return list in graph.
    Description: Support calculate gradient for graph with list return.
    Expectation: No exception.
    """
    @jit
    def foo():
        x = sorted((5, 3, 1, 4, 2))
        return x
    assert list(foo()) == [1, 2, 3, 4, 5]
