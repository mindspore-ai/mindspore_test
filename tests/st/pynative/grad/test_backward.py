# Copyright 2020-2023 Huawei Technologies Co., Ltd
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
""" test_auto_grad """

import numpy as np
import pytest
import mindspore
from mindspore import _Function
from mindspore import Tensor, Parameter, jit, _no_grad
from mindspore.common.api import _pynative_executor
from mindspore.ops import stop_gradient
from mindspore._c_expression import run_backward
from tests.mark_utils import arg_mark

_pynative_executor.set_grad_flag(True)


@property
def requires_grad(self):
    """
    Return whether the parameter requires gradient.
    """
    return self._requires_grad


@requires_grad.setter
def requires_grad(self, value=True):
    if not isinstance(value, bool):
        raise TypeError("The argument `requires_grad` must be bool type")
    self._requires_grad = value


@property
def grad(self):
    """
    Return whether the parameter requires gradient.
    """
    return self._grad


@grad.setter
def grad(self, value):
    self._grad = value


@property
def is_leaf(self):
    """
    Return whether the parameter requires gradient.
    """
    return self._is_leaf


def backward(self, gradient=None, retain_graph=None, create_graph=False, inputs=None):
    return run_backward((self,), gradient, retain_graph, create_graph, inputs, allow_unreachable=True,
                        accumulate_grad=True)


Tensor.requires_grad = requires_grad
Tensor.grad = grad
Tensor.backward = backward
Tensor.is_leaf = is_leaf


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_normal():
    """
    Feature: Test backward normal
    Description: Test backward api
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad = True
    y = Tensor([5], mindspore.float32)
    y.requires_grad = True
    z = x * y
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([5], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_normal2():
    """
    Feature: Test backward api
    Description: Test backward api requires_grad
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    y = Tensor([5], mindspore.float32)
    y.requires_grad_()
    z = x * y
    h = stop_gradient(z)
    h.requires_grad_()
    z = h * h
    z += h
    z.backward()
    assert np.allclose(h.grad.asnumpy(), np.array([31], dtype=np.float32), 0.00001, 0.00001)
    assert x.grad is None
    assert y.grad is None


@jit
def jit_net(x):
    y = x * x
    z = y + y
    return z


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_with_jit():
    """
    Feature: Test backward api
    Description: Test backward api requires_grad
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    y = Tensor([5], mindspore.float32)
    y.requires_grad_()
    z = x * y
    h = jit_net(z)
    z = h * h
    z += h
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([270300], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_accumulate():
    """
    Feature: Test backward api
    Description: Test backward api accumulate
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    y = Tensor([5], mindspore.float32)
    y.requires_grad_()
    z = x * y
    z.backward()
    h = x + y
    h.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([6], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_accumulate():
    """
    Feature: Test backward api
    Description: Test backward api accumulate
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    y = Tensor([5], mindspore.float32)
    y.requires_grad_()
    z = x * y
    z.backward()
    h = x + y
    h.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([6], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_constant_tensor():
    """
    Feature: Test backward api
    Description: Test backward api accumulate
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    y = Tensor([5], mindspore.float32)
    y.requires_grad_()
    with _no_grad():
        z = x * y
    with pytest.raises(RuntimeError) as err:
        z.backward()
    assert "The output tensor you provided doesn't requires grad" in str(err.value)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_set_grad_none():
    """
    Feature: Test backward api
    Description: Test backward api set grad none
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    y = Tensor([5], mindspore.float32)
    y.requires_grad_()
    z = x * y
    z.backward()
    x.grad = None
    y.grad = None
    h = x + y
    h.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_set_grad_err_shape():
    """
    Feature: Test backward api
    Description: Test backward api set grad none
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    y = Tensor([5], mindspore.float32)
    y.requires_grad_()
    z = x * y
    z.backward()
    with pytest.raises(RuntimeError) as err:
        x.grad = Tensor([3], mindspore.int64)
    assert "The grad dtype and shape should be same" in str(err.value)
    with pytest.raises(RuntimeError) as err:
        y.grad = Tensor([3, 6], mindspore.float32)
    assert "The grad dtype and shape should be same" in str(err.value)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_twice_exception():
    """
    Feature: Test backward api
    Description: Test backward api set grad none
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    y = Tensor([5], mindspore.float32)
    y.requires_grad_()
    z = x * y
    z.backward()
    with pytest.raises(RuntimeError) as err:
        z.backward()
    assert "Try to backward the graph twice" in str(err.value)


class FunctionNet(_Function):
    @staticmethod
    def forward(ctx, x):
        t = x * x
        z = t + t
        return z

    @staticmethod
    def backward(ctx, z):
        return z * 3


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_twice_custom_function_exception():
    """
    Feature: Test backward api
    Description: Test backward api set grad none
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    y = Tensor([5], mindspore.float32)
    y.requires_grad_()
    z = x * y
    z = FunctionNet.apply(z)
    z.backward()
    with pytest.raises(RuntimeError) as err:
        z.backward()
    assert "Try to backward the graph twice" in str(err.value)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_twice():
    """
    Feature: Test backward api
    Description: Test backward api set grad none
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    y = Tensor([5], mindspore.float32)
    y.requires_grad_()
    z = x * y
    z.backward(retain_graph=True)
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([10], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(y.grad.asnumpy(), np.array([6], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_high_order():
    """
    Feature: Test backward api
    Description: Test backward api high order
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    z = x * x
    z.backward(create_graph=True)
    x.grad.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([8.], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_register_hook():
    """
    Feature: Test backward api
    Description: Test backward api register hook
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    x.register_hook(lambda grad_out: grad_out * 2)
    z = x * x
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([12.], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_set_requires_grad_register_hook():
    """
    Feature: Test backward api
    Description: Test backward api set requires grad false and register hook
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    x.requires_grad_()
    x.register_hook(lambda grad_out: grad_out * 2)
    x.requires_grad = False
    x.requires_grad = True
    z = x * x
    z.backward()
    assert np.allclose(x.grad.asnumpy(), np.array([12.], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_backward_is_leaf():
    """
    Feature: Test backward api
    Description: Test backward api is leaf
    Expectation: Success.
    """
    x = Tensor([3], mindspore.float32)
    assert x.is_leaf is True
    x.requires_grad_()
    z = x * x
    assert z.is_leaf is False

