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
""" test graph mode sequence getitem """

import numpy as np
from mindspore import context, jit, mutable
from mindspore.common.tensor import Tensor
from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tuple_getitem_with_constant_bool_index():
    """
    Feature: sequence getitem with bool index
    Description: sequence getitem
    Expectation: No exception
    """
    @jit
    def foo():
        m = (1, 2, 3, 4)
        return m[True], m[False]

    ret1, ret2 = foo()
    assert ret1 == 2
    assert ret2 == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tuple_getitem_with_constant_bool_index_2():
    """
    Feature: sequence getitem with bool index
    Description: sequence getitem
    Expectation: No exception
    """
    @jit
    def foo(x):
        m = (x, x+1, x+2, x+3)
        return m[True], m[False]

    ret1, ret2 = foo(Tensor([1, 2, 3, 4]))
    assert np.all(ret1.asnumpy() == np.array([2, 3, 4, 5]))
    assert np.all(ret2.asnumpy() == np.array([1, 2, 3, 4]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tuple_getitem_with_variable_bool_index():
    """
    Feature: sequence getitem with bool index
    Description: sequence getitem
    Expectation: No exception
    """
    @jit
    def foo(x):
        if x > 0:  # pylint: disable=simplifiable-if-statement
            index = True
        else:
            index = False
        m = (x, x+1, x+2, x+3)
        return m[index], m[not index]

    ret1, ret2 = foo(Tensor([1]))
    assert np.all(ret1.asnumpy() == np.array([2]))
    assert np.all(ret2.asnumpy() == np.array([1]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tuple_getitem_with_variable_bool_index_2():
    """
    Feature: sequence getitem with bool index
    Description: sequence getitem
    Expectation: No exception
    """
    @jit(backend="ms_backend")
    def foo(x, a):
        m = (x, x+1, x+2, x+3)
        return m[a == 1], m[a == 2]

    ret1, ret2 = foo(Tensor([1]), mutable(1))
    assert np.all(ret1.asnumpy() == np.array([2]))
    assert np.all(ret2.asnumpy() == np.array([1]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_tuple_getitem_with_variable_bool_index_3():
    """
    Feature: sequence getitem with bool index
    Description: sequence getitem
    Expectation: No exception
    """
    @jit(backend="ms_backend")
    def foo(a):
        m = (1, 2, 3, 4)
        return m[a == 1], m[a == 2]

    ret1, ret2 = foo(mutable(1))
    assert ret1 == 2
    assert ret2 == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_list_getitem_with_constant_bool_index():
    """
    Feature: sequence getitem with bool index
    Description: sequence getitem
    Expectation: No exception
    """
    @jit
    def foo():
        m = [1, 2, 3, 4]
        return m[True], m[False]

    ret1, ret2 = foo()
    assert ret1 == 2
    assert ret2 == 1


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_list_getitem_with_constant_bool_index_2():
    """
    Feature: sequence getitem with bool index
    Description: sequence getitem
    Expectation: No exception
    """
    @jit(backend="ms_backend")
    def foo(x):
        m = [x, x+1, x+2, x+3]
        return m[True], m[False]

    ret1, ret2 = foo(Tensor([1, 2, 3, 4]))
    assert np.all(ret1.asnumpy() == np.array([2, 3, 4, 5]))
    assert np.all(ret2.asnumpy() == np.array([1, 2, 3, 4]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_list_getitem_with_variable_bool_index():
    """
    Feature: sequence getitem with bool index
    Description: sequence getitem
    Expectation: No exception
    """
    @jit
    def foo(x):
        if x > 0:  # pylint: disable=simplifiable-if-statement
            index = True
        else:
            index = False
        m = [x, x+1, x+2, x+3]
        return m[index], m[not index]

    ret1, ret2 = foo(Tensor([1]))
    assert np.all(ret1.asnumpy() == np.array([2]))
    assert np.all(ret2.asnumpy() == np.array([1]))


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_list_getitem_with_variable_bool_index_2():
    """
    Feature: sequence getitem with bool index
    Description: sequence getitem
    Expectation: No exception
    """
    @jit(backend="ms_backend")
    def foo(x, a):
        m = [x, x+1, x+2, x+3]
        return m[a == 1], m[a == 2]

    ret1, ret2 = foo(Tensor([1]), mutable(1))
    assert np.all(ret1.asnumpy() == np.array([2]))
    assert np.all(ret2.asnumpy() == np.array([1]))
