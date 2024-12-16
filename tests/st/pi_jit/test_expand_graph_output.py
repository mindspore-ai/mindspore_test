import pytest
from mindspore import jit, context, Tensor, JitConfig
from mindspore.common import dtype as mstype
from mindspore._c_expression import get_code_extra
from .share.utils import match_value, assert_executed_by_graph_mode
from tests.mark_utils import arg_mark

def run_call_result(x, y):
    ret = (x, y)
    for index in range(22):
        ret = (ret, x * index, y * index)
    return ret

def run_nested_tuple(x, y):
    res = run_call_result(x, y)
    return x, y, (x * 10, y * 10), ((x * 20, y * 20), (x * 30, y * 30), res)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_call_result])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
def test_call_result(func, x, y):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_first_input with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    wrapped_func = jit(func, mode='PIJit')
    ms_res = wrapped_func(x, y)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(x, y)
    match_value(ms_res, res)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [run_nested_tuple])
@pytest.mark.parametrize('x', [Tensor([[1., 2.], [3., 4.]], mstype.float32)])
@pytest.mark.parametrize('y', [Tensor([[5., 6.], [7., 8.]], mstype.float32)])
def test_nested_tuple(func, x, y):
    """
    Feature: ALL TO ALL
    Description: test cases for test_run_grad_first_input with hook
    Expectation: the result match
    Note: Must call pijit first, the args x and y will be modified in pynative
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    wrapped_func = jit(func, mode='PIJit')
    ms_res = wrapped_func(x, y)
    assert_executed_by_graph_mode(wrapped_func)
    res = func(x, y)
    match_value(ms_res, res)
