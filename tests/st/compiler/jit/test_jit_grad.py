# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
from mindspore.common import Tensor
from mindspore import context, jit
from mindspore.ops.composite import GradOperation
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_tuple():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(x, y):
        return 2 * x[0] + y

    def func(x, y):
        x = x * 3
        return inner_func((x,), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_tuple_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(m):
        return 2 * m[0][0] + m[1]

    def func(x, y):
        x = x * 3
        return inner_func(((x,), y))

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_list():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(x, y):
        return 2 * x[0] + y

    def func(x, y):
        x = x * 3
        return inner_func([x,], y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_list_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(m):
        return 2 * m[0][0] + m[1]

    def func(x, y):
        x = x * 3
        return inner_func([[x,], y])

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_dict():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(m):
        return 2 * m["x"] + m["y"]

    def func(x, y):
        x = x * 3
        return inner_func({"x": x, "y": y})

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_dict_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(m):
        return 2 * m["x"][0] + m["y"]

    def func(x, y):
        x = x * 3
        return inner_func({"x": (x,), "y": y})

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_vargs():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(*args):
        return 2 * args[0][0] + args[1]

    def func(x, y):
        x = x * 3
        return inner_func((x,), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_vargs_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(*args):
        return 2 * args[0][0][0] + args[1]

    def func(x, y):
        x = x * 3
        return inner_func(((x,),), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@pytest.mark.skip(reason="Jit handle kwargs with mutable error, fix later")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_kwargs():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(**kwargs):
        return 2 * kwargs["m"][0] + kwargs["n"]

    def func(x, y):
        x = x * 3
        return inner_func(m=(x,), n=y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@pytest.mark.skip(reason="Jit handle kwargs with mutable error, fix later")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_kwargs_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(**kwargs):
        return 2 * kwargs["m"][0][0] + kwargs["n"]

    def func(x, y):
        x = x * 3
        return inner_func(m=([x,],), n=y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([6, 6, 6]))


@pytest.mark.skip(reason="Jit handle kwargs with mutable error, fix later")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_grad_tensor_in_sequence_with_kwargs_3():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: success.
    """
    @jit
    def inner_func(**kwargs):
        return 2 * kwargs["m"][0] + kwargs["n"]

    def func(x, y):
        return inner_func(m=mutable((x,)), n=y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])
    ret = GradOperation()(func)(a, b)
    assert np.all(ret.asnumpy() == np.array([2, 2, 2]))


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_invalid_input():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: RuntimeError.
    """
    @jit
    def inner_func(m):
        return 2 * m["x"][0] + m["y"]

    def func(x, y):
        x = x * 3
        return inner_func({"x": (x, "a"), "y": y})

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])

    with pytest.raises(RuntimeError) as error_info:
        GradOperation()(func)(a, b)
    assert "contains tensor with gradient but can not mutable" in str(error_info.value)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_jit_grad_with_invalid_input_2():
    """
    Feature: Test grad scene for tensor in container used as jit input.
    Description: Test grad scene for tensor in container used as jit input.
    Expectation: RuntimeError.
    """
    @jit
    def inner_func(x, y):
        return 2 * x[0] + y

    def func(x, y):
        x = x * 3
        return inner_func((x, None), y)

    context.set_context(mode=context.PYNATIVE_MODE)
    a = Tensor([1, 2, 3])
    b = Tensor([1, 1, 1])

    with pytest.raises(RuntimeError) as error_info:
        GradOperation()(func)(a, b)
    assert "contains tensor with gradient but can not mutable" in str(error_info.value)
