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
# pylint: disable=unused-variable
import pytest
import numpy as np

import mindspore as ms
from mindspore import ops
from mindspore.ops.operations._inner_ops import AntiQuant


def test_primitive_init_keyword_argument():
    """
    Feature: DynamicShape.
    Description: Test init keyword argument.
    Expectation: No exception.
    """
    @ms.jit
    def func1(x, arg):
        return ops.AvgPool(arg, pad_mode="VALID", strides=arg)(x)

    @ms.jit
    def func2(x, arg):
        return ops.AvgPool(arg, pad_mode="VALID", strides=arg)(x=x)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    func1(x, 1)
    func1(x, ms.mutable(1))
    func2(x, 1)
    func2(x, ms.mutable(1))


def test_primitive_keyword_argument():
    """
    Feature: DynamicShape.
    Description: Test keyword argument
    Expectation: No exception.
    """
    @ms.jit
    def func1(x, axis, keep_dims):
        return ops.ReduceSum(keep_dims=keep_dims)(x=x, axis=axis)

    @ms.jit
    def func2(x, keep_dims):
        return ops.ReduceSum(keep_dims=keep_dims)(x)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    func1(x, 1, True)
    func1(x, ms.mutable(1), False)
    func1(x, (1, 2), ms.mutable(True))
    func1(x, ms.mutable((1, 2)), ms.mutable(False))
    func2(x, True)
    func2(x, ms.mutable(True))


def test_primitive_call_keyword_argument_in_different_order():
    """
    Feature: DynamicShape.
    Description: Test keyword arguments in different orders.
    Expectation: No exception.
    """
    @ms.jit
    def func1(x, num_groups, weight, bias, eps):
        return ops.auto_generate.GroupNorm()(x, weight=weight, bias=bias, num_groups=num_groups, eps=eps)

    @ms.jit
    def func2(x, num_groups, weight, bias, eps):
        return ops.auto_generate.GroupNorm()(x, num_groups, weight, bias=bias, eps=eps)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    num_groups = 2
    weight = ms.Tensor(np.random.rand(36,).astype(np.float32))
    bias = ms.Tensor(np.random.rand(36,).astype(np.float32))
    eps = 1e-5

    func1(x, num_groups, weight, bias, eps)
    func1(x, ms.mutable(2), weight, bias, ms.mutable(1e-5))

    func2(x, num_groups, weight, bias, eps)
    func2(x, ms.mutable(2), weight, bias, ms.mutable(1e-5))


def test_primitive_call_keyword_argument_concat():
    """
    Feature: DynamicShape.
    Description: Test keyword argument of ops.Concat().
    Expectation: Raise TypeError.
    """
    @ms.jit
    def func(x, axis):
        return ops.Concat(axis=axis)(tensors=x)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x1 = ms.Tensor(np.random.rand(2, 2).astype(np.float32))
    x2 = ms.Tensor(np.random.rand(2, 2).astype(np.float32))

    func((x1, x2), 1)
    func((x1, x2), ms.mutable(1))


def test_primitive_call_keyword_argument_with_wrong_keyword():
    """
    Feature: DynamicShape.
    Description: Test keyword argument with wrong keyword.
    Expectation: Raise TypeError.
    """
    @ms.jit
    def func(x, axis):
        return ops.Softmax(axis)(x=x)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    with pytest.raises(RuntimeError) as info1:
        func(x, (-1,))
    assert "Got an unexpected keyword argument" in str(info1.value)
    with pytest.raises(RuntimeError) as info2:
        func(x, ms.mutable((-1,)))
    assert "Got an unexpected keyword argument" in str(info2.value)


def test_primitive_call_keyword_argument_batchnorm():
    """
    Feature: DynamicShape.
    Description: Test keyword argument of ops.BatchNorm()
    Expectation: No exception.
    """
    @ms.jit
    def func(x, mean, var):
        return ms.ops.BatchNorm()(input_x=x, scale=mean, bias=var, mean=mean, variance=var)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.random.rand(10, 36, 12, 12).astype(np.float32))
    mean = ms.Tensor(np.random.rand(36,).astype(np.float32))
    var = ms.Tensor(np.random.rand(36,).astype(np.float32))
    func(x, mean, var)


def test_primtive_antiquant_with_kwargs():
    """
    Feature: DynamicShape.
    Description: Test keyword argument of ops.AntiQuant()
    Expectation: Raise Exception.
    """
    @ms.jit
    def func(x, scale, offset):
        return AntiQuant()(x=x, scale=scale, offset=offset)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor([50.0, 20.0], ms.int8)
    with pytest.raises(TypeError) as info1:
        func(x, 1.0, 2.0)
    assert "But got input argument" in str(info1.value)
    with pytest.raises(TypeError) as info2:
        func(x, ms.mutable(1.0), 2.0)
    assert "But got input argument" in str(info2.value)


def test_partial_keyword_argument():
    """
    Feature: DynamicShape.
    Description: Test keyword argument of ops.Partial()
    Expectation: No exception.
    """
    @ms.jit
    def func1(x, axis, keep_dims):
        return ms.ops.Partial()(ops.ReduceSum(keep_dims=keep_dims), axis=axis)(x=x)

    @ms.jit
    def func2(x, axis, keep_dims):
        return ms.ops.Partial()(ops.ReduceSum(keep_dims=keep_dims), axis=axis)(x)

    ms.set_context(mode=ms.GRAPH_MODE, jit_syntax_level=ms.STRICT)
    x = ms.Tensor(np.random.rand(2, 3, 4, 5).astype(np.float32))
    func1(x, (-1,), True)
    func1(x, ms.mutable(-1,), True)

    func2(x, (-1,), True)
    func2(x, ms.mutable(-1,), True)
