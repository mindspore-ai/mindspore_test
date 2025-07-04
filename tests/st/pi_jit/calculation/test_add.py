import pytest 
from mindspore import numpy as np
from mindspore import Tensor, jit, context
from ..share.utils import match_array
from tests.mark_utils import arg_mark


@jit(capture_mode="bytecode")
def add(a, b):
    return a + b


@jit
def jit_add(a, b):
    return a + b


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [add])
@pytest.mark.parametrize('ms_func', [jit_add])
@pytest.mark.parametrize('test_data', [(1, 100)])
def test_add_int(func, ms_func, test_data):
    """
    Feature:
        Addition Operation Across Different Data Types

    Description:
        Evaluate the addition operation for various data types (e.g., integers, floats, strings)
        using specified functions.

    Expectation:
        Computed result should match the expected result for each data type.
        No errors should occur during execution.
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = a + b
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [add])
@pytest.mark.parametrize('ms_func', [jit_add])
@pytest.mark.parametrize('test_data', [(1.0, 100.0)])
def test_add_float(func, ms_func, test_data):
    """
    Feature:
        Addition Operation Across Different Data Types

    Description:
        Evaluate the addition operation for various data types (e.g., integers, floats, strings)
        using specified functions.

    Expectation:
        Computed result should match the expected result for each data type.
        No errors should occur during execution.
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = a + b
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [add])
@pytest.mark.parametrize('ms_func', [jit_add])
@pytest.mark.parametrize('test_data', [(2.0, Tensor(np.ones((2, 3)).astype(np.float32)))])
def test_add_float_tensor(func, ms_func, test_data):
    """
    Feature:
        Addition Operation Across Different Data Types

    Description:
        Evaluate the addition operation for various data types (e.g., integers, floats, strings)
        using specified functions.

    Expectation:
        Computed result should match the expected result for each data type.
        No errors should occur during execution.
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = a + b
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [add])
@pytest.mark.parametrize('ms_func', [jit_add])
@pytest.mark.parametrize('test_data', [(Tensor(np.ones((2, 3)).astype(np.float32)),
                                        Tensor(np.ones((2, 3)).astype(np.float32)))])
def test_add_tensor(func, ms_func, test_data):
    """
    Feature:
        Addition Operation Across Different Data Types

    Description:
        Evaluate the addition operation for various data types (e.g., integers, floats, strings)
        using specified functions.

    Expectation:
        Computed result should match the expected result for each data type.
        No errors should occur during execution.
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = a + b
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [add])
@pytest.mark.parametrize('ms_func', [jit_add])
@pytest.mark.parametrize('test_data', [((1, 2, 3), (4, 5, 6))])
def test_add_tuple(func, ms_func, test_data):
    """
    Feature:
        Addition Operation Across Different Data Types

    Description:
        Evaluate the addition operation for various data types (e.g., integers, floats, strings)
        using specified functions.

    Expectation:
        Computed result should match the expected result for each data type.
        No errors should occur during execution.
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = a + b
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [add])
@pytest.mark.parametrize('ms_func', [jit_add])
@pytest.mark.parametrize('test_data', [("Hello", "World")])
def test_add_str(func, ms_func, test_data):
    """
    Feature:
        Addition Operation Across Different Data Types

    Description:
        Evaluate the addition operation for various data types (e.g., integers, floats, strings)
        using specified functions.

    Expectation:
        Computed result should match the expected result for each data type.
        No errors should occur during execution.
    """
    a, b = test_data
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(a, b)
    context.set_context(mode=context.GRAPH_MODE)
    ms_res = ms_res = ms_func(a, b)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
