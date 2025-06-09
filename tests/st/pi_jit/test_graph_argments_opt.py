import pytest
from mindspore import jit, context, Tensor
from mindspore.nn import Cell
from mindspore.common import dtype as mstype
from mindspore._c_expression import update_pijit_default_config
from .share.utils import match_array, match_value, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import pi_jit_with_config
from tests.st.pi_jit.one_stage.test_utils import save_graph_ir, check_ir_num

# set feature expand graph input on
config = { "expand_graph_input": True, "eliminate_redundant_args":True}

def run_sequence(s):
    return s[0] + s[3]

def run_sequence_append(s):
    s1 = []
    s1.append(s[0])
    s1.append(s[1])
    s1.append(s[2])
    s1.append(s[3])
    return s1[0] + s1[3]

def run_sequence_assign(s):
    s1 = [None, None, None, None]
    s1[0] = s[0]
    s1[1] = s[1]
    s1[2] = s[2]
    s1[3] = s[3]
    return s1[0] + s1[3]

def run_sequence_insert(s):
    s1 = []
    s1.insert(0, s[0])
    s1.insert(1, s[1])
    s1.insert(2, s[2])
    s1.insert(3, s[3])
    return s1[0] + s1[3]

def run_sequence_len(s):
    return s[0] + s[1] + len(s)

def run_sequence_pop(s):
    s.pop()
    s.pop()
    return s[0] + s[1]

def run_sequence_reverse(s):
    s.reverse()
    return s[0] + s[3]

def run_nested_sequence(s):
    s = s[0] + s[1]
    return s[0] + s[1]

def run_dict(d):
    return d.get('a') + d['d']

def run_dict_keys_values(d):
    keys = tuple(d.keys())
    values = tuple(d.values())
    d1 = {}
    for idx in range(len(keys)):
        d1[keys[idx]] = values[idx]
    return d1.get('a') + d1['d']

def run_dict_dict(d):
    d1 = dict([[key, d[key]] for key in d.keys()])
    return d1.get('a') + d1['d']

def make_dict_by_key_value(d):
    keys = tuple(d.keys())
    values = tuple(d.values())
    d1 = {}
    for idx in range(len(keys)):
        d1[keys[idx]] = values[idx]
    return d1

def make_dict_by_dict(d):
    return dict([[key, d[key]] for key in d.keys()])

def make_dict_by_items(d):
    return dict([[key, value] for key, value in d.items()])

def run_dict_sub_func(d):
    d1 = make_dict_by_dict(make_dict_by_key_value(d))
    return d1.get('a') + d1['d']

def run_dict_items(d):
    d1 = dict([[key, value] for key, value in d.items()])
    return d1.get('a') + d1['d']

def run_dict_len(d):
    return d.get('a') + d['d'] + len(d)

def run_dict_update(d):
    d1 = d.get('d1')
    d1.update(d['d2'])
    return {'ret' : d1['a'] + d1.get('d') }

def run_nested_dict(d):
    d1 = make_dict_by_key_value(d.get('d1'))
    d2 = make_dict_by_dict(d['d2'])
    return {'ret' : d1['a'] + d2.get('d') }

def run_nested_dict_mix(d):
    d1 = make_dict_by_dict(d.get('d1'))
    d2 = make_dict_by_items(d['d2'])
    return {'ret' : d1['a'] + d2.get('d') }

def run_mix_case(s, d):
    d1 = make_dict_by_dict(d.get('d1'))
    d2 = make_dict_by_items(d['d2'])
    return s[0][0] + d1.get('a') + d2['b'][0]

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_sequence, run_sequence_append, run_sequence_assign, run_sequence_len, run_sequence_reverse])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[100., 200.], [300., 400.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[1000., 2000.], [3000., 4000.]], mstype.float32)])
def test_list(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand list in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = [x1, x2, y1, y2]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    s[2] = x1
    s[3] = x2
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_sequence_pop, run_sequence_insert])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[100., 200.], [300., 400.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[1000., 2000.], [3000., 4000.]], mstype.float32)])
def test_list_break(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand list in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = [x1, x2, y1, y2]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    s1 = [x1, x2, y1, y2]
    res = func(s1)
    match_value(s, s1, error=0, err_msg=str(s))
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_nested_sequence])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_nest_list(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand nested list in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = [[x1, x2], [y1, y2]]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_sequence, run_sequence_append, run_sequence_assign, run_sequence_len])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_tuple(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand tuple in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = (x1, x2, y1, y2)
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_nested_sequence])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_nested_tuple(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand nested tuple in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = ((x1, x2), (y1, y2))
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_dict, run_dict_keys_values, run_dict_dict, run_dict_sub_func, run_dict_items, run_dict_len])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_dict(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand dict in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    d = {'a': x1, 'b': x2, 'c': y1, 'd':y2}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_nested_dict, run_nested_dict_mix])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_nested_dict(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand nested tuple in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    d = { 'd1': {'a': x1, 'b': x2}, 'd2': {'c': y1, 'd':y2}}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(d)
    match_array(res['ret'], ms_res['ret'], error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_dict_update])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_nested_dict_update(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand nested tuple in parameter
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    d = { 'd1': {'a': x1, 'b': x2}, 'd2': {'c': y1, 'd':y2}}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(d)
    res = func(d)
    match_array(res['ret'], ms_res['ret'], error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_mix_case])
@pytest.mark.parametrize('x1', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('x2', [Tensor([[10., 20.], [30., 40.]], mstype.float32)])
@pytest.mark.parametrize('y1', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
@pytest.mark.parametrize('y2', [Tensor([[50., 60.], [70., 80.]], mstype.float32)])
def test_mix_case(func, x1, x2, y1, y2):
    """
    Feature: ALL TO ALL
    Description: test cases for expand vargs in function
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    s = [[1, x1], [3, 4, 5]]
    d = {'d1': {'a' : y1, 'b' : 1}, 'd2': {'a' : 8, 'b':[3, 4, 5]}}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s, d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s, d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    s = [[1, x1], [6, 7, 8, 9, 10]]
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s, d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s, d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
    d = {'d1': {'a' : y1, 'b' : 1}, 'd2': {'a' : 8, 'b':[6, 7, 8]}}
    wrapped_func = pi_jit_with_config(func, jit_config=config)
    ms_res = wrapped_func(s, d)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(s, d)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_elimination_01():
    """
    Feature: parameter elimination
    Description: test cases for parameter elimination
    Expectation: The result match and no exception
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.x = 1

        @jit(capture_mode="bytecode", backend="ms_backend")
        def construct(self, x):
            if self.x > 3:
                self.x = x
            else:
                self.x = self.x + 1
            return self.x + x

    update_pijit_default_config(eliminate_redundant_args=True)
    net = Net()
    expected_y = [2, 4, 6, 6, 8, 10, 12, 14, 16, 18]
    actual_y = []
    for i in range(10):
        x = Tensor([i])
        y = net(x)
        actual_y.append(y.asnumpy()[0])

    assert actual_y == expected_y
    check_ir_num('graph_before_compile', 8)

@save_graph_ir(ir_name='graph_before_compile')
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_parameter_elimination_02():
    """
    Feature: parameter elimination
    Description: test cases for parameter elimination
    Expectation: The result match and no exception
    """
    class Net(Cell):
        def __init__(self):
            super().__init__()
            self.x = 1

        @jit(capture_mode="bytecode", backend="ms_backend")
        def construct(self, x):
            if self.x > 3:
                self.x = x
            else:
                self.x = self.x + 1
            return self.x + x

    update_pijit_default_config(eliminate_redundant_args=False)
    net = Net()
    expected_y = [2, 4, 6, 6, 8, 10, 12, 14, 16, 18]
    actual_y = []
    for i in range(10):
        x = Tensor([i])
        y = net(x)
        actual_y.append(y.asnumpy()[0])

    assert actual_y == expected_y
    check_ir_num('graph_before_compile', 8)
