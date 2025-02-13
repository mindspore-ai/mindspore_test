# Copyright 2023 Huawei Technologies Co., Ltd
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

''' test resolve of side effect in pijit , by break_count_ judge is support side effect handing'''

from dataclasses import dataclass
import pytest
from mindspore import jit, Tensor, context, ops
from mindspore.nn import Cell, ReLU
from mindspore._c_expression import get_code_extra
import dis
import mindspore
import types
import numpy
from tests.mark_utils import arg_mark
from ..share.utils import match_array, assert_executed_by_graph_mode
from tests.st.pi_jit.share.utils import pi_jit_with_config

def assert_no_graph_break(func, call_count: int = None):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    if call_count is not None:
        assert jcr['code']['call_count_'] == call_count


def assert_graph_break(func, break_count: int = 1):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    assert jcr is not None
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == break_count


class NetAssign0002(Cell):

    def __init__(self):
        super().__init__()
        self.relu = ReLU()

    def construct(self, x, y):
        x[1] = y
        return x


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_store_subscr_side_effect_1():
    """
    Feature: STORE SUBSCR + HAS_ARGS
    Description: wipe out graph_break in store subscr has args
    Expectation: no exception
    """

    def func(x):
        x[0] = Tensor([1, 2])
        x[1] = Tensor([1, 2])
        return x

    jit(function=func, capture_mode="bytecode")([Tensor([1]), Tensor([1])])
    jcr = get_code_extra(func)
    new_code = jcr["code"]["compiled_code_"]
    for i in dis.get_instructions(new_code):
        if i.opname == "STORE_SUBSCR":
            flag = True
    assert flag
    context.set_context(mode=context.PYNATIVE_MODE)

    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_store_subscr_side_effect_2():
    """
    Feature: STORE_SUBSCR + NO_ARGS + OPERATION
    Description: wipe out graph_break in store subscr no args
    Expectation: no exception
    """

    def func():
        x = [Tensor([1]), Tensor([1])]
        x[0] = Tensor([1, 2])
        return x

    jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_del_subscr_side_effect_3():
    """
    Feature: DEL_SUBSCR + NO_ARGS + OPERATION
    Description: wipe out graph_break in store subscr no args
    Expectation: no exception
    """

    def func(arg):
        del arg[0]
        return arg

    jit(function=func, capture_mode="bytecode")([Tensor([1]), Tensor([1])])
    jcr = get_code_extra(func)
    new_code = jcr["code"]["compiled_code_"]

    for i in dis.get_instructions(new_code):
        if i.opname == "DELETE_SUBSCR":
            flag = True
    assert flag
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dict_pop_side_effect_4():
    """
    Feature: DICT POP side effect
    Description: wipe out graph_break in dict pop no args
    Expectation: no exception
    """

    def func():
        d = {"a": Tensor([1, 2]), "b": Tensor([1, 2])}
        d.pop("b")
        return d

    jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_dict_pop_side_effect_5():
    """
    Feature: DICT POP side effect 2
    Description: wipe out graph_break in dict pop as args
    Expectation: no exception
    """

    def func(d):
        d.pop("b")
        return d

    jit(function=func, capture_mode="bytecode")({"a": Tensor([1, 2]), "b": Tensor([1, 2])})
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_store_global_side_effect_6():
    """
    Feature: STORE_GLOBAL
    Description: wipe out graph_break in store global no args
    Expectation: no exception
    """

    def func():
        global tmp
        tmp = Tensor([1])
        tmp *= 2
        return tmp

    jit(function=func, capture_mode="bytecode")()
    jcr = get_code_extra(func)
    context.set_context(mode=context.PYNATIVE_MODE)
    assert jcr["break_count_"] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_del_global_side_effect_7():
    """
    Feature: DEL GLOBAL side effect
    Description: wipe out graph_break in dict pop no args
    Expectation: NameError
    """

    def func():
        global tmp
        tmp = Tensor([1])
        tmp *= 2
        del tmp
        return tmp

    with pytest.raises(NameError, match="name 'tmp' is not defined"):
        jit(function=func, capture_mode="bytecode")()

    context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_fix_bug_store_subscr_side_effect_1():
    """
    Feature: STORE SUBSCR + FIX BUGS
    Description: wipe out graph_break in store subscr has args
    Expectation: no exception
    """

    def func(net):
        x = [Tensor([1, 2]), Tensor([2, 3])]
        y = Tensor([5, 6])
        net(x, y)
        return x

    net = NetAssign0002()
    result = jit(function=func, capture_mode="bytecode")(net)
    jcr = get_code_extra(func)

    assert jcr["break_count_"] == 0
    assert (result[1] == Tensor([5, 6])).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('test_optimize', [True, False])
@pytest.mark.skip(reason="fix later, error because of bytecode reorder")
def test_modify_mix1(test_optimize):
    """
    Feature: Side-effect handle
    Description: Test list append, list set item, dict set item
    Expectation: No exception
    """

    def func(arg):
        x = []
        y = {}
        y['z'] = 1  # not need track, same as `y = {'z' : 1}`
        x.append(y)  # not need track, same as `x = [y]`
        y['x'] = x  # must be record, y is referenced by x
        x.append(y)  # must be record, x is referenced by y
        y['y'] = y  # must be record, make dict can't do this
        res = arg + x[-1]['z']
        if test_optimize:
            return res  # no side_effect
        return y, res

    excepted = func(Tensor([1]))
    result = jit(function=func, capture_mode="bytecode")(Tensor([1]))

    assert str(excepted) == str(result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="Need Fixed")
def test_modify_mix2():
    """
    Feature: Side-effect handle
    Description: Test dict.pop, delete dict item
    Expectation: No exception
    """

    def func(x):
        x['a'] = 0
        y = {}
        y['b'] = x
        res = y['b']['param'] + x.pop('param')
        del x['remove']
        return res

    x1 = {'param': Tensor([1]), 'remove': 1}
    x2 = {**x1}
    excepted = func(x1)
    result = jit(function=func, capture_mode="bytecode")(x2)

    assert excepted == result
    assert x1 == x2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_global_modified_cross_module():
    """
    Feature: Side-effect handle
    Description: Test global modify with different modules
    Expectation: No exception
    """
    global magic_number
    global new_func

    def func(x):
        global magic_number
        y = new_func(x)
        magic_number = x
        return x + y + magic_number

    global_dict = mindspore.__dict__
    new_func = types.FunctionType(func.__code__, global_dict)
    global_dict['new_func'] = int

    x = Tensor([1])
    excepted = func(x)
    magic_number_excepted = global_dict.pop('magic_number')

    del magic_number
    result = jit(function=func, capture_mode="bytecode")(x)
    magic_number_result = global_dict.pop('magic_number')

    assert x == magic_number_excepted == magic_number_result
    assert Tensor([5]) == result == excepted


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_object_consistency():
    """
    Feature: Test side-effect
    Description: Test the modification of same object from multiple source
    Expectation: No exception
    """

    @jit(capture_mode="bytecode")
    def object_consistency(x, y):
        x.f = y.get
        y.test = x
        y.list.append(1)
        test = x.y.test.f()  # recognize x.y is y
        x.y.list[0] = 0
        return test

    def get():
        return test_object_consistency

    x = object_consistency
    y = test_object_consistency
    y.get = get
    y.list = []
    x.y = y
    res = object_consistency(x, y)
    assert res is test_object_consistency
    assert y.list[0] == 0 and y.test is x and x.f is y.get


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="Need Fixed")
def test_object_consistency2():
    """
    Feature: Test side-effect
    Description: Test the modification of same object from multiple source
    Expectation: No exception
    """

    @jit(capture_mode="bytecode")
    def func(x, y):
        x.append(1)
        y.append(2)
        return x[0] + x[1] + y[1]

    a = Tensor([1])
    l = [a]
    res1 = func(l, l)
    res2 = func(l, [l])

    assert res1 == 3
    assert res2 == 4
    assert l == [a, 1, 2, 1]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("assign_fn", [Tensor.assign_value,
                                       mindspore.ops.assign,
                                       lambda x, v: x.__setitem__(slice(None), v)
                                       ])
def test_tensor_assign(assign_fn):
    """
    Feature: Test side-effect
    Description: Test side-effect rollback. Test side-effect restore
    Expectation: No exception
    """

    @jit(capture_mode="bytecode")
    def func(x, y, assign, rand):
        a = x + y
        if rand:
            # break at here, test side-effect rollback
            rand = rand(2)
            y = y + 1
        x = assign(x, y)
        b = x + y
        return a, b, rand

    x = mindspore.Parameter(Tensor([1]), name="x")
    y = mindspore.Parameter(Tensor([2]), name="y")
    a1, b1, rand1 = func(x, y, assign_fn, None)
    a2, b2, rand2 = func(x, y, assign_fn, numpy.random.rand)

    assert x.value() == Tensor(3)
    assert a1 == Tensor([3]) and b1 == Tensor([4])
    assert a2 == Tensor([4]) and b2 == Tensor([6])
    assert rand1 is None and isinstance(rand2, numpy.ndarray)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("assign_fn", [Tensor.assign_value,
                                       mindspore.ops.assign,
                                       lambda x, v: x.__setitem__(slice(None), v)
                                       ])
def test_tensor_consistency(assign_fn):
    """
    Feature: Test side-effect
    Description: Test the modification of same object from multiple source. Test no return value side-effect
    Expectation: No exception
    """
    if assign_fn is Tensor.assign_value:
        pytest.skip("MakeTensorCopy, at do call, parameter handle, func graph builder use parameter as constant value")

    if assign_fn.__name__ == "<lambda>":
        pytest.skip("sub graph side-effect value can't return to top graph")

    @jit(capture_mode="bytecode")
    def func(assign, x, y, x1, y1):
        a = x + y
        assign(x, y)
        assign(x, x1)
        assign(y, y1)
        b = x + y
        return a, b

    x = mindspore.Parameter(Tensor([1]), name="x")
    y = mindspore.Parameter(Tensor([2]), name="y")
    x1 = Tensor([3])
    y1 = Tensor([4])
    a1, b1 = func(assign_fn, x, y, x1, y1)
    a2, b2 = func(assign_fn, x, y, x1, y1)

    assert x.value() == x1 and y.value() == y1
    assert a1 == Tensor([3]) and b1 == Tensor([7])
    assert a2 == Tensor([7]) and b2 == Tensor([7])


jit_cfg = {'compile_with_try': False}


@pytest.mark.skip(reason='unsupported for now')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_side_effect_of_dict():
    """
    Feature: Side-effect handle
    Description: Test dict set item
    Expectation: No exception
    """

    def fn2(d: dict):
        d['x'] *= 2
        return d['x'] + 1

    def fn(d: dict):
        x = d['x']
        d['x'] = x + 1
        y = fn2(d)
        return d['x'] + y

    x1 = {'x': Tensor([1, 2, 3])}
    o1 = fn(x1)

    x2 = {'x': Tensor([1, 2, 3])}
    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x2)

    match_array(o1, o2)
    assert len(x1) == len(x2)
    match_array(x1['x'], x2['x'])
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0


class Net1(mindspore.nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.memory = Tensor([1, 2, 3])

    @pi_jit_with_config(jit_config=jit_cfg)
    def construct(self, x: Tensor):
        self.update_memory()
        return x + self.memory

    def update_memory(self):
        self.memory = self.memory + 1
        return self.memory


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_side_effect_of_store_attr():
    """
    Feature: Side-effect handle
    Description: Test STORE_ATTR
    Expectation: No exception
    """
    net = Net1()
    x = Tensor([1, 1, 1])
    o = net(x)

    match_array(o, Tensor([3, 4, 5]))
    match_array(net.memory, Tensor([2, 3, 4]))
    jcr = get_code_extra(net.construct.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0


class Net2(mindspore.nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.memory_a = Tensor([1, 2, 3])
        self.memory_b = Tensor([1, 1, 1])

    def construct(self, x: Tensor):
        a = self.memory_a * 2
        self.memory_a += 1
        c = self.memory_a + self.memory_b
        memory = self.update_memory(x)
        return a + c + memory

    def update_memory(self, x: Tensor):
        self.memory_a = self.memory_a + x
        self.memory_b = self.memory_b * 2
        self.update_memory_once_more()
        return self.memory_a + self.memory_b

    def update_memory_once_more(self):
        self.memory_a = self.memory_a + self.memory_b
        self.memory_b = self.memory_b * 2
        return self.memory_a


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_side_effect_of_store_attr_in_deep_subgraph():
    """
    Feature: Side-effect handle
    Description: Test STORE_ATTR
    Expectation: No exception
    """
    net1 = Net2()
    x = Tensor([1, 1, 1])
    o1 = net1(x)

    net2 = Net2()
    fn = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    o2 = fn(x)

    match_array(o1, o2)
    match_array(net1.memory_a, net2.memory_a)
    match_array(net1.memory_b, net2.memory_b)
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0


class Net3(mindspore.nn.Cell):
    def __init__(self):
        super(Net3, self).__init__()
        self.net = Net2()

    def construct(self, x: Tensor):
        y = self.net(x)
        return x + y


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_side_effect_of_store_attr_in_nested_cell():
    """
    Feature: Side-effect handle
    Description: Test STORE_ATTR
    Expectation: No exception
    """
    net1 = Net3()
    x = Tensor([1, 1, 1])
    o1 = net1(x)

    net2 = Net3()
    fn = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    o2 = fn(x)

    match_array(o1, o2)
    match_array(net1.net.memory_a, net2.net.memory_a)
    match_array(net1.net.memory_b, net2.net.memory_b)
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0


class Net4(mindspore.nn.Cell):
    def __init__(self):
        super(Net4, self).__init__()
        self.memory = Tensor([1, 2, 3])

    @pi_jit_with_config(jit_config=jit_cfg)
    def construct(self, x: Tensor):
        self.update_memory()
        return 2 * x

    def update_memory(self):
        self.memory = self.memory + 1


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_has_side_effect_but_has_no_return_value():
    """
    Feature: Side-effect handle
    Description: Test a function that has no return statement
    Expectation: No exception
    """
    net1 = Net4()
    x = Tensor([1, 1, 1])
    o1 = net1(x)

    net2 = Net4()
    fn = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    o2 = fn(x)

    match_array(o1, o2)
    match_array(net1.memory, net2.memory)
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_setitem():
    """
    Feature: Side-effect handle
    Description: Test list setitem
    Expectation: No exception
    """

    def fn(x: Tensor, axis: int):
        lst = [1] * len(x.shape)
        lst[axis + 1] = -1
        return lst

    x = ops.ones((2, 3, 4))
    o1 = fn(x, 0)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, 0)

    assert o1 == o2
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_setitem_by_inplace_add():
    """
    Feature: Side-effect handle
    Description: Test list setitem by inplace operation
    Expectation: No exception
    """

    def fn(lst: list, i: int, x: Tensor):
        lst[i - 1] += ops.abs(x)
        return lst

    lst = [Tensor([1, 1, 1]), Tensor([2, 2, 2]), Tensor([3, 3, 3])]
    x = Tensor([-1, 0, 1])
    o1 = fn(lst, 2, x)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    lst = [Tensor([1, 1, 1]), Tensor([2, 2, 2]), Tensor([3, 3, 3])]
    o2 = fn(lst, 2, x)

    assert o2 is lst
    assert len(o1) == len(o2)
    for l, r in zip(o1, o2):
        match_array(l, r)
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0


@pytest.mark.skip(reason='unsupported for now')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_list_setitem_in_subgraph():
    """
    Feature: Side-effect handle
    Description: Test list setitem
    Expectation: No exception
    """

    def fn2(lst: list):
        lst[0] *= 2
        return lst[0] + 2

    def fn(lst: list):
        x = lst[0]
        lst[0] += 1
        y = fn2(lst)
        return lst[0] + y + x

    x1 = [Tensor([1, 2, 3])]
    o1 = fn(x1)

    x2 = [Tensor([1, 2, 3])]
    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x2)

    match_array(o1, o2)
    assert len(x1) == len(x2)
    match_array(x1[0], x2[0])
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0


@pytest.mark.skip(reason='unsupported')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_setitem_by_slice():
    """
    Feature: Side-effect handle
    Description: Test Tensor setitem by slice
    Expectation: No exception
    """

    def fn(x: Tensor, indices: tuple, y: int):
        x[:, indices] = y
        return x

    x = ops.ones((2, 3, 4))
    indices = (0, 1)
    o1 = fn(x, indices, -1)

    fn = pi_jit_with_config(fn, jit_config=jit_cfg)
    o2 = fn(x, indices, -1)

    match_array(o1, o2)
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0


class Net5(mindspore.nn.Cell):
    def __init__(self):
        super(Net5, self).__init__()
        self.a = 0

    def construct(self, x: Tensor):
        return x + self.a


@pytest.mark.skip('unsupported')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_graph_reusing_for_store_int_attr_outside_of_jit():
    """
    Feature: Side-effect handle
    Description: Test graph reusing
    Expectation: No exception
    """
    net1 = Net5()
    out1 = []
    x = Tensor([1, 2, 3])
    for _ in range(3):
        out1.append(net1(x))
        net1.a += 1

    net2 = Net5()
    fn = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    phase = ''
    for i in range(3):
        o = fn(x)
        match_array(o, out1[i])
        assert_executed_by_graph_mode(fn)
        jcr = get_code_extra(fn.__wrapped__)
        if i == 0:
            phase = jcr['phase_']
        else:
            assert jcr['phase_'] == phase  # reusing the first graph
        assert jcr['call_count_'] == (i + 1)


class Net6(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = None

    def construct(self, x: Tensor):
        i = self.inner(x)
        return ops.mul(x, i)

    def inner(self, x: Tensor):
        self.a = self.create_tuple(x)
        return len(self.a)

    def create_tuple(self, x: Tensor):
        return ops.add(x, 1), ops.sub(x, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_return_const_value_and_has_tuple_side_effect():
    """
    Feature: Side-effect handle
    Description: 1.subgraph has side effect and side effect node is a tuple; 2.subgraph return const value.
    Expectation: No exception
    """
    net1 = Net6()
    x = Tensor([1, 2, 3])
    o1 = net1(x)

    net2 = Net6()
    net2.construct = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    o2 = net2(x)

    match_array(o1, o2)
    assert type(net1.a) is tuple and type(net2.a) is tuple
    assert len(net1.a) == 2 and len(net2.a) == 2
    match_array(net1.a[0], net2.a[0])
    match_array(net1.a[1], net2.a[1])
    assert_no_graph_break(net2.construct)


@dataclass
class MyData:
    id: int
    tensor: Tensor


class Net7(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = None

    def construct(self, x: Tensor):
        data = self.inner(x)
        print('graph break', end='')
        return data.id * data.tensor

    def inner(self, x: Tensor):
        self.a, b = self.create_tuple(x)
        return MyData(len(b), b)

    def create_tuple(self, x: Tensor):
        return ops.add(x, 1), ops.sub(x, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_return_user_defined_class_and_has_side_effect():
    """
    Feature: Side-effect handle
    Description: 1.subgraph has side effect; 2.subgraph return user defined class.
    Expectation: No exception
    """
    net1 = Net7()
    x = Tensor([1, 2, 3])
    o1 = net1(x)

    net2 = Net7()
    net2.construct = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    o2 = net2(x)

    match_array(o1, o2)
    match_array(net1.a, net2.a)
    assert_graph_break(net2.construct)


class Net8(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = 0

    def construct(self, x: Tensor):
        x = ops.add(x, 1)
        y = self.inner(x)
        return x + y

    def inner(self, x: Tensor):
        a, b = self.create_tuple(x)
        self.a += a  # because of subgraph break, should reset the side-effect operation in subgraph
        print('graph break', end='')
        return b * 2

    def create_tuple(self, x: Tensor):
        return ops.add(x, 1), ops.sub(x, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_and_reset_side_effect_node_1():
    """
    Feature: Side-effect handle
    Description: 1.subgraph has side effect; 2.subgraph has graph break.
    Expectation: No exception
    """
    net1 = Net8()
    x = Tensor([1, 2, 3])
    o1 = net1(x)

    net2 = Net8()
    net2.construct = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    o2 = net2(x)

    match_array(o1, o2)
    match_array(net1.a, net2.a)
    assert_graph_break(net2.construct)


class Net9(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = 0

    def construct(self, x: Tensor):
        x = ops.add(x, 1)
        y = self.inner(x)
        return x + y

    def inner(self, x: Tensor):
        self.a += ops.add(x, 1)  # because of subgraph break, should reset the side-effect operation in subgraph
        b = ops.sub(x, 1)
        print('graph break', end='')
        return b * 2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_and_reset_side_effect_node_2():
    """
    Feature: Side-effect handle
    Description: 1.subgraph has side effect; 2.subgraph has graph break.
    Expectation: No exception
    """
    net1 = Net9()
    x = Tensor([1, 2, 3])
    o1 = net1(x)

    net2 = Net9()
    net2.construct = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    o2 = net2(x)

    match_array(o1, o2)
    match_array(net1.a, net2.a)
    assert_graph_break(net2.construct)


class Net10(mindspore.nn.Cell):
    def __init__(self):
        super().__init__()
        self.a = 0

    def construct(self, x: Tensor):
        self.a += x
        x = ops.add(x, 1)
        y = self.inner(x)
        return x + y

    def inner(self, x: Tensor):
        self.a += ops.add(x, 1)  # because of subgraph break, should reset the side-effect operation in subgraph
        b = self.inner_with_side_effect_1(x)
        print('graph break', end='')
        return b * 2

    def inner_with_side_effect_1(self, x: Tensor):
        self.a += x  # because of subgraph break, should reset the side-effect operation in subgraph
        return self.inner_with_side_effect_2(x)

    def inner_with_side_effect_2(self, x: Tensor):
        self.a += (2 * x)  # because of subgraph break, should reset the side-effect operation in subgraph
        return ops.sub(x, 1)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_subgraph_break_and_reset_side_effect_node_3():
    """
    Feature: Side-effect handle
    Description: 1.subgraph has side effect; 2.subgraph has graph break.
    Expectation: No exception
    """
    net1 = Net10()
    x = Tensor([1, 2, 3])
    o1 = net1(x)

    net2 = Net10()
    net2.construct = pi_jit_with_config(net2.construct, jit_config=jit_cfg)
    o2 = net2(x)

    match_array(o1, o2)
    match_array(net1.a, net2.a)
    assert_graph_break(net2.construct)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("has_side_effect", [False, True])
def test_side_effect_eliminate(has_side_effect : bool):
    """
    Feature: Side-effect handle
    Description: Validate side effect optimize
    Expectation: No exception
    """
    if not has_side_effect:
        pytest.skip("Variable escape analysis not implement. For this case, the variable 't' is escaped if 'r[1]' returned")

    @jit(capture_mode='bytecode')
    def func(has_side_effect : bool, x : Tensor = Tensor([3])):
        t = Tensor([1])
        r = [[t + t], [t]]
        t[:] = x
        return r[has_side_effect]

    excepted = func.__wrapped__(has_side_effect)
    result = func(has_side_effect)
    assert excepted == result
    if not has_side_effect:
        assert_executed_by_graph_mode(func)
    else:
        assert_no_graph_break(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_side_effect_eliminate_2():
    """
    Feature: Side-effect handle
    Description: Validate side effect optimize
    Expectation: No exception
    """
    @jit(capture_mode='bytecode')
    def func(x : Tensor = Tensor([3, 3])):
        # t = Tensor.new_zeros(x, x.shape) builtin method new_zeros of PyCapsule
        # t = mindspore.tensor(x) fix it after Tensor constant
        t = ops.add(x, x)
        t[:1] = 1
        t = t + x
        t[1:] = 1
        return t + x

    excepted = func.__wrapped__()
    result = func()
    assert (excepted == result).all()
    assert_executed_by_graph_mode(func)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_side_effect_merge():
    """
    Feature: Side-effect merge
    Description: Validate side effect merge. Result must be correct
    Expectation: No exception
    """
    def func(x):
        x[:1]=1
        x[2:]=2
        return x

    data = numpy.random.rand(4, 4)
    x1=Tensor(data)
    x2=Tensor(data)

    func(x1)
    jit(func, capture_mode="bytecode")(x2)
    assert (x1 == x2).all()

    assert_no_graph_break(func)
    jcr = get_code_extra(func)
    opnames = [i.opname for i in dis.get_instructions(jcr['code']['compiled_code_'])]
    assert opnames.count('STORE_SUBSCR') == 1
