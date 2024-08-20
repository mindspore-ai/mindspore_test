import pytest
import sys
from mindspore import Tensor, jit, context
from .share.utils import match_array
from tests.mark_utils import arg_mark

@pytest.fixture(autouse=True)
def skip_if_python_version_too_high():
    if sys.version_info >= (3, 9) or sys.version_info < (3, 8):
        pytest.skip("Only tests on Python 3.8.")

@jit(mode="PIJit")
def build_list_unpack(x, y, z):
    return [*x, *y, *z]

def ms_build_list_unpack(x, y, z):
    return [*x, *y, *z]

@jit(mode="PIJit")
def build_set_unpack(x, y, z):
    return {*x, *y, *z}

def ms_build_set_unpack(x, y, z):
    return {*x, *y, *z}

def build_tuple(a, b, c, d, e, f, g, h, i):
    return (a, b, c, d, e, f, g, h, i)

@jit(mode="PIJit")
def build_tuple_unpack(x, y, z):
    return (*x, *y, *z)

def ms_build_tuple_unpack(x, y, z):
    return (*x, *y, *z)

@jit(mode="PIJit")
def build_tuple_unpack_with_call(x, y, z):
    return build_tuple(*x, *y, *z)

def ms_build_tuple_unpack_with_call(x, y, z):
    return build_tuple(*x, *y, *z)

@jit(mode="PIJit")
def build_map_unpack(x, y, z):
    return {**x, **y, **z}

def ms_build_map_unpack(x, y, z):
    return {**x, **y, **z}

@jit(mode="PIJit")
def build_map_unpack_with_call(x, y, z):
    return build_tuple(**x, **y, **z)

def ms_build_map_unpack_with_call(x, y, z):
    return build_tuple(**x, **y, **z)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [build_list_unpack])
@pytest.mark.parametrize('ms_func', [ms_build_list_unpack])
@pytest.mark.parametrize('x', [(1, 2, 3)])
@pytest.mark.parametrize('y', [(5, 6)])
@pytest.mark.parametrize('z', [(7, 8, 9, 10)])
def test_build_list_unpack(func, ms_func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_LIST_UNPACK in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    ms_res = ms_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [build_set_unpack])
@pytest.mark.parametrize('ms_func', [ms_build_set_unpack])
@pytest.mark.parametrize('x', [(1, 2, 3)])
@pytest.mark.parametrize('y', [(5, 6)])
@pytest.mark.parametrize('z', [(7, 8, 9, 10)])
def test_build_set_unpack(func, ms_func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_SET_UNPACK in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = set(func(x, y, z))
    ms_res = ms_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [build_tuple_unpack])
@pytest.mark.parametrize('ms_func', [ms_build_tuple_unpack])
@pytest.mark.parametrize('x', [(1, 2, 3)])
@pytest.mark.parametrize('y', [(5, 6)])
@pytest.mark.parametrize('z', [(7, 8, 9, 10)])
def test_build_tuple_unpack(func, ms_func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_TUPLE_UNPACK in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    ms_res = ms_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [build_tuple_unpack_with_call])
@pytest.mark.parametrize('ms_func', [ms_build_tuple_unpack_with_call])
@pytest.mark.parametrize('x', [(1, 2, 3)])
@pytest.mark.parametrize('y', [(5, 6)])
@pytest.mark.parametrize('z', [(7, 8, 9, 10)])
def test_build_tuple_unpack_with_call(func, ms_func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_TUPLE_UNPACK_WITH_CALL in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    ms_res = ms_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [build_map_unpack])
@pytest.mark.parametrize('ms_func', [ms_build_map_unpack])
@pytest.mark.parametrize('x', [{"a":1, "b":2, "c":3}])
@pytest.mark.parametrize('y', [{"d":5, "e":6}])
@pytest.mark.parametrize('z', [{"f":7, "g":8, "h":9, "i":10}])
def test_build_map_unpack(func, ms_func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_MAP_UNPACK in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    ms_res = ms_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('func', [build_map_unpack_with_call])
@pytest.mark.parametrize('ms_func', [ms_build_map_unpack_with_call])
@pytest.mark.parametrize('x', [{"a":1, "b":2, "c":3}])
@pytest.mark.parametrize('y', [{"d":5, "e":6}])
@pytest.mark.parametrize('z', [{"f":7, "g":8, "h":9, "i":10}])
def test_build_map_unpack_with_call(func, ms_func, x, y, z):
    """
    Feature: ALL TO ALL
    Description: test cases for BUILD_MAP_UNPACK_WITH_CALL in 3.8
    Expectation: the result match
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    res = func(x, y, z)
    ms_res = ms_func(x, y, z)
    match_array(res, ms_res, error=0, err_msg=str(ms_res))
