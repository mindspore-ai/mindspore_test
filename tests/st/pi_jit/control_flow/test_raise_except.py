from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore import jit
from mindspore._c_expression import get_code_extra
import dis
from tests.mark_utils import arg_mark
import sys
import pytest


@pytest.fixture(autouse=True)
def skip_if_python_version_too_high():
    if sys.version_info >= (3, 11):
        pytest.skip("Skipping tests on Python 3.11 and higher.")


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_1():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        try:
            raise Exception("new exception in try!")
        finally:
            pass

    with pytest.raises(Exception, match="new exception in try!"):
        jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_2():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        try:
            try:
                try:
                    raise Exception("new exception in try!")
                finally:
                    pass
            finally:
                pass
        finally:
            pass

    with pytest.raises(Exception, match="new exception in try!"):
        jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_3():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        try:
            try:
                try:
                    raise Exception("new exception in try!")
                finally:
                    pass
            finally:
                raise Exception("new exception2 in finally!")
        finally:
            pass

    with pytest.raises(Exception, match="new exception2 in finally!"):
        jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_4():
    """
    Feature: Test with try/except/finally
    Description: Test with raise exception process
    Expectation: no except raise.
    """

    def func():
        try:
            try:
                try:
                    raise Exception("new exception in try!")
                except ArithmeticError:
                    pass
                except BufferError:
                    pass
                except Exception:
                    pass
                finally:
                    pass
            finally:
                pass
        finally:
            pass

    jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_5():
    """
    Feature: Test with try/except/finally
    Description: Test with raise exception process
    Expectation: resolve except correctly.
    """

    def func():
        try:
            try:
                try:
                    raise Exception("new exception in try!")
                except ArithmeticError:
                    pass
                except BufferError:
                    pass
                except Exception:
                    raise Exception("new exception2 in except!")
                finally:
                    pass
            finally:
                raise Exception("new exception3 in finally!")
        finally:
            pass

    with pytest.raises(Exception, match="new exception3 in finally!"):
        jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_6():
    """
    Feature: Test with try/except/finally
    Description: Test with raise exception process
    Expectation: resolve except correctly,do not support raise except from callee.
    """

    def func():
        try:
            try:
                try:
                    raise Exception("new exception in try!")
                except ArithmeticError:
                    pass
                except BufferError:
                    pass
                except Exception:
                    raise Exception("new exception2 in except!")
                finally:
                    pass
            finally:
                raise Exception("new exception3 in finally!")
        finally:
            pass

    def func2():
        func()

    with pytest.raises(Exception, match="new exception3 in finally!"):
        jit(fn=func2, mode="PIJit")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


def test_except_case_7():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        i = 1
        try:
            i = 2
            raise Exception("new exception in try!")
        except Exception:
            i = 3
            pass
        finally:
            pass
        return i

    got = jit(fn=func, mode="PIJit")()
    expected = func()
    print(got)
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


def test_except_case_8():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        i = 1
        try:
            i = 2
            raise Exception("new exception in try!")
        except:
            i = 3
            raise Exception("new exception in except!")
        finally:
            pass
        return i

    with pytest.raises(Exception, match="new exception in except!"):
        got = jit(fn=func, mode="PIJit")()
        expected = func()
        assert got == expected

    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


def test_except_case_9():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        i = 1
        try:
            i = 2
            raise Exception
        except Exception:
            i = 3
        finally:
            pass
        return i

    got = jit(fn=func, mode="PIJit")()
    expected = func()
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


def test_except_case_10():
    """
    Feature: Test with raise exception
    Description: Test with raise exception process
    Expectation: raise exception.
    """

    def func():
        i = 1
        try:
            i = 2
        except Exception:
            i = 3
        else:
            i = 4
        return i

    got = jit(fn=func, mode="PIJit")()
    expected = func()
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_11():
    """
    Feature: Test with try/except/finally
    Description: Test with raise exception process
    Expectation: resolve except correctly.
    """

    def func():
        try:
            try:
                try:
                    raise BufferError
                except ArithmeticError:
                    pass
                except BufferError:
                    raise BufferError("new BufferError in except!")
                except Exception:
                    raise Exception("new exception2 in except!")
                finally:
                    pass
            finally:
                pass
        finally:
            pass

    with pytest.raises(Exception, match="new BufferError in except!"):
        jit(fn=func, mode="PIJit")()
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_12():
    """
    Feature: Test try/finally
    Description: Test with try/finally process
    Expectation: break count == 0.
    """

    def func():
        try:
            return 1
        finally:
            i = 3
        return 2

    got = jit(fn=func, mode="PIJit")()
    expected = func()
    assert got == expected
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_13():
    """
    Feature: Test try/finally
    Description: Test with try/finally process
    Expectation: break count == 1.
    """

    def func():
        try:
            i = 1
            raise ArithmeticError
        except AssertionError:
            i = 2
            return i
        else:
            i = 3
        finally:
            i = 4
        return 5

    with pytest.raises(ArithmeticError):
        jit(fn=func, mode="PIJit")()

    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 1


@arg_mark(
    plat_marks=["cpu_linux"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_except_case_14():
    """
    Feature: Test with/try/finally
    Description: Test with/try/finally process
    Expectation: break count == 0.
    """

    class MyFile:
        def __init__(self):
            pass

        def __enter__(self):
            pass

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def func():
        with MyFile():
            try:
                raise ArithmeticError
            except ArithmeticError:
                i = 2
            return i

    got = jit(fn=func, mode="PIJit")()
    assert got == 2
    jcr = get_code_extra(func)
    assert jcr["break_count_"] == 0
