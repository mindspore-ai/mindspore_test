import pytest
from mindspore import Tensor, jit, ops
from mindspore import numpy as np
import mindspore.nn as nn
from mindspore import context
from .share.utils import match_array
from tests.mark_utils import arg_mark
from mindspore._c_expression import get_code_extra

condition = not (sys.version_info.major == 3 and sys.version_info.minor in [8,9])

jit_config={'recapture_loop_body':True}

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test001():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        result = a
        for i in range(10):
            result = result + a
            if result[0,0] > 10:
                result = result + a
        return result
    fn = jit(func, mode="PIJit", jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test002():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        result = a
        if a.shape[0] == 2:
            for i in range(10):
                result = result + a
                if result[0,0] > 10:
                    result = result + a
        else:
            result = result + 1
        return result
    fn = jit(func, mode="PIJit", jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test003():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        result = a
        for i in range(10):
            result = result + a
            if result[0,0] > 10:
                result = result + a
            else:
                result = result + 2
        return result
    fn = jit(func, mode="PIJit", jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test004():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        result = a
        def fn():
            return a
        for i in range(10):
            result = result + a
            if result[0,0] > 10:
                result = result + a
            else:
                result = result + 2
        return result
    fn = jit(func, mode="PIJit", jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skip(reason="RunGraph Failed !!!")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test005():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        result = a
        for i in range(10):
            def fn():
                return i
            result = result + a
            if result[0,0] > 10:
                result = result + a
            else:
                result = result + 2
        return result
    fn = jit(func, mode="PIJit", jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test006():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(tensor_list):
        result = tensor_list[0]
        for i in tensor_list:
            result = result + i
            if i[0,0] > 1:
                result = result + 1
        return result
    x1 = np.randn((2,4))
    x2 = np.randn((2,4))
    x3 = np.randn((2,4))
    x4 = np.randn((2,4))
    tensor_list=[x1,x2,x3,x4]
    fn = jit(func, mode="PIJit", jit_config=jit_config)
    expect = func(tensor_list)
    got = fn(tensor_list)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test007():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(tensor_list):
        result = 0
        for i in tensor_list:
            result = result + i.sum()
            if i[0,0] > 1:
                result = result + 1
        return result
    x1 = np.randn((2,4))
    x2 = np.randn((3,4))
    x3 = np.randn((4,3))
    x4 = np.randn((5,3))
    tensor_list=[x1,x2,x3,x4]
    fn = jit(func, mode="PIJit", jit_config=jit_config)
    expect = func(tensor_list)
    got = fn(tensor_list)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose

@pytest.mark.skipif(condition, reason="Only support python 3.8, 3.9")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test008():
    """
    Feature: ALL TO ALL
    Description: test cases for args support in PYNATIVE mode
    Expectation: the result match
    TEST_SUMMARY: test __call__ function in class
    """
    def func(a):
        i = 0
        result = a
        while(i < 10):
            if result[0,0] > 1:
                result = result + 1
            else:
                result = result + 2
            i += 1
        return result
    fn = jit(func, mode="PIJit", jit_config=jit_config)
    x1 = np.randn((2,4))
    expect = func(x1)
    got = fn(x1)
    allclose = np.isclose(expect,got).all()
    jcr = get_code_extra(fn.__wrapped__)
    assert jcr['stat'] == 'GRAPH_CALLABLE'
    assert jcr['break_count_'] == 0
    assert allclose