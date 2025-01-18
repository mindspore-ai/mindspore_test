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
"""mint module."""
from __future__ import absolute_import
import mindspore.ops as ops
from mindspore.ops.primitive import constexpr
from mindspore.common.tensor import Tensor
from mindspore.ops.function.array_func import gather_ext as gather
from mindspore.ops.function.nn_func import conv2d_ext as conv2d
from mindspore.mint.nn.functional import sigmoid
from mindspore.mint.nn import functional
from mindspore.mint import linalg
from mindspore.mint import special
from mindspore.mint import distributed
from mindspore.ops import erf
from mindspore.ops.function.math_func import linspace_ext as linspace
from mindspore.ops.function.math_func import median_ext as median
from mindspore.ops.function.array_func import ones_like_ext as ones_like
from mindspore.ops.function.array_func import full_ext as full
from mindspore.ops.function.array_func import zeros_like_ext as zeros_like
from mindspore.ops.function.array_func import unique_ext as unique
from mindspore.ops.function.array_func import chunk_ext as chunk
from mindspore.ops.functional_overload import empty
from mindspore.ops.function.array_func import empty_like
from mindspore.ops.function.math_func import isclose
from mindspore.ops.auto_generate import abs
from mindspore.ops.auto_generate import clone
from mindspore.ops.function.array_func import full_like_ext as full_like
# 1
from mindspore.ops.function.math_func import divide
from mindspore.ops.auto_generate import topk_ext as topk
from mindspore.ops.function.math_func import roll
# 2
from mindspore.ops.function.math_func import sin
# 3
from mindspore.ops.functional_overload import clamp, where
from mindspore.ops.functional_overload import clip
from mindspore.ops.functional_overload import fmod
from mindspore.ops.functional_overload import max
from mindspore.ops.functional_overload import min
# 4
from mindspore.ops.auto_generate import sinc
from mindspore.ops.auto_generate import sinh
from mindspore.ops.auto_generate import cosh
from mindspore.ops.function.math_func import xlogy_ext as xlogy
# 5
from mindspore.ops.auto_generate import cumsum_ext as cumsum
# 6
from mindspore.ops.auto_generate import stack_ext as stack

# 7
from mindspore.ops.function.array_func import unsqueeze
# 8
from mindspore.ops.auto_generate import transpose_ext as transpose
from mindspore.ops.auto_generate import batch_norm_elemt
from mindspore.ops.auto_generate import batch_norm_gather_stats_with_counts
from mindspore.ops.auto_generate import batch_norm_stats
# 9
from mindspore.ops.auto_generate import masked_select
from mindspore.ops.function.math_func import cross
# 10
from mindspore.ops.function.math_func import ne
# 11
from mindspore.ops.function.math_func import cdist as cdist_
# 12
from mindspore.ops.functional_overload import repeat_interleave
# 13
from mindspore.ops.functional import flip
# 14
from mindspore.ops.auto_generate import mv
# 15
from mindspore.ops.auto_generate import flatten_ext as flatten
# 16
from mindspore.ops.functional import matmul
from mindspore.ops.auto_generate import bmm_ext as bmm
# 17

# 18

# 19
from mindspore.ops.functional import log
# 20

# 21
from mindspore.ops.function.math_func import mul_ext as mul
# 22
from mindspore.ops.functional import cumprod
# 23
from mindspore.ops.auto_generate import exp2
# 24

# 25
from mindspore.ops.functional import greater, gt
# 26
from mindspore.ops.functional import eq
# 27
from mindspore.ops.functional import reciprocal
# 28
from mindspore.ops.functional import exp
# 29

# 30
from mindspore.ops.functional import searchsorted
# 31

# 32

# 33

# 34

# 35
from mindspore.ops.functional import erfinv
# 36

# 37
from mindspore.ops.function.array_func import nonzero
# 38

# 39

# 40

# 41

# 42
from mindspore.ops.function.math_func import argmax_ext as argmax
# 43

# 44
from mindspore.ops.functional import cos
# 45

# 46
from mindspore.ops.function.math_func import bitwise_and_ext as bitwise_and
# 47
from mindspore.ops.function.math_func import bitwise_or_ext as bitwise_or
# 48
from mindspore.ops.function.math_func import bitwise_xor_ext as bitwise_xor
# 49
from mindspore.ops.function.math_func import baddbmm_ext as baddbmm
# 50
from mindspore.ops.functional import tile
# 51

# 52

# 53

# 54
from mindspore.ops.function.random_func import normal_ext as normal
# 55

# 56
from mindspore.ops.function.math_func import norm_ext as norm
# 57
from mindspore.ops.functional import broadcast_to
# 58
from mindspore.ops.function.math_func import greater_equal
# 59
from mindspore.ops.functional import square
# 60

# 61
from mindspore.ops.functional import rsqrt
# 62
from mindspore.ops.functional import maximum
# 63
from mindspore.ops.functional import minimum
# 64
from mindspore.ops.functional import ravel
# 65
from mindspore.ops.functional import logical_and
# 66
from mindspore.ops.functional import logical_not
# 67
from mindspore.ops.functional import logical_or
# 68
from mindspore.ops.functional import logical_xor
# 69
from mindspore.ops.functional import less_equal, le
# 70
from mindspore.ops.functional import negative, neg
# 71

# 72

# 73
from mindspore.ops.functional import ceil
# 74
from mindspore.ops.function.array_func import sort_ext as sort
# 75
from mindspore.ops.functional import less, lt
# 76
from mindspore.ops.function.math_func import pow_ext as pow
# 77

# 78
from mindspore.ops.function import arange_ext as arange
# 79

# 80
from mindspore.ops.functional_overload import div
# 81
from mindspore.ops.auto_generate import index_select_ext as index_select
# 82
from mindspore.ops.auto_generate import cummin_ext as cummin
# 83
from mindspore.ops.auto_generate import narrow
# 84

# 85
from mindspore.mint import nn, optim
# 86

# 87
from mindspore.ops.auto_generate import trunc
# 88

# 89
from mindspore.ops.auto_generate import argsort_ext as argsort
# 90
from mindspore.ops.auto_generate import isinf
# 91

# 92
from mindspore.ops.function.math_func import polar
# 93

# 94
from mindspore.ops.function.math_func import tanh
# 95
from mindspore.ops.function.math_func import diff_ext as diff
# 96

# 97

# 98

# 99

# 100

# 101

# 102

# 103

# 104

# 105

# 106

# 107

# 108

# 109
from mindspore.ops.auto_generate import argmin_ext as argmin
# 110
from mindspore.ops.function.nn_func import softmax_ext
# 111

# 112

# 113

# 114

# 115

# 116

# 117

# 118

# 119

# 120
from mindspore.ops.auto_generate import isneginf_ext as isneginf
# 121

# 122

# 123
from mindspore.ops.function.math_func import var_ext as var

# 151
from mindspore.ops.function.math_func import acos_ext as acos
from mindspore.ops.function.math_func import arccos_ext as arccos
# 152
from mindspore.ops.function.math_func import acosh_ext as acosh
from mindspore.ops.function.math_func import arccosh_ext as arccosh
# 172
from mindspore.ops.function.math_func import asin_ext as asin
from mindspore.ops.function.math_func import arcsin_ext as arcsin
# 173
from mindspore.ops.function.math_func import asinh_ext as asinh
from mindspore.ops.function.math_func import arcsinh_ext as arcsinh
# 174
from mindspore.ops.function.math_func import atan_ext as atan
from mindspore.ops.function.math_func import arctan_ext as arctan
# 175
from mindspore.ops.function.math_func import atanh
from mindspore.ops.function.math_func import arctanh
# 176
from mindspore.ops.function.math_func import atan2_ext as atan2
from mindspore.ops.function.math_func import arctan2_ext as arctan2

# 177
from mindspore.ops.function.math_func import round

# 182
from mindspore.ops.function.math_func import bernoulli_ext as bernoulli

# 204
from mindspore.ops.auto_generate import erfc
# 207
from mindspore.ops.auto_generate import expm1
# 208
from mindspore.ops.function.array_func import eye
from mindspore.ops.function.random_func import randperm_ext as randperm
from mindspore.ops.function.random_func import rand_ext as rand
from mindspore.ops.function.random_func import rand_like_ext as rand_like
from mindspore.ops.function.random_func import randn_ext as randn
from mindspore.ops.function.random_func import randn_like_ext as randn_like
from mindspore.ops.function.random_func import randint_ext as randint
from mindspore.ops.function.random_func import randint_like_ext as randint_like
# 210
from mindspore.ops.auto_generate import floor
# 231
from mindspore.ops.function.math_func import inverse_ext as inverse
# 239
from mindspore.ops.functional_overload import lerp
# 244
from mindspore.ops.auto_generate import log1p
# 261
from mindspore.ops.function.random_func import multinomial_ext as multinomial
# 275
from mindspore.ops.functional_overload import remainder
# 285
from mindspore.ops.function.array_func import scatter_add_ext as scatter_add
# 289
from mindspore.ops.auto_generate import sign

from mindspore.ops.auto_generate import select_ext as select

# 301
from mindspore.ops.function.math_func import tan

# 303
from mindspore.ops.auto_generate import trace_ext as trace
from mindspore.ops.auto_generate import gcd

from mindspore.ops.function.array_func import reshape

from mindspore.ops.auto_generate import outer_ext as outer

# 304
from mindspore.ops.function.array_func import tril_ext as tril
# 520
from mindspore.ops.function.math_func import bincount_ext as bincount

# 305
from mindspore.ops import triu

# 308
from mindspore.ops.auto_generate import mm_ext as mm

# 382
from mindspore.ops.function.math_func import dstack

# 501
from mindspore.ops.function.math_func import addbmm_ext as addbmm

# 502
from mindspore.ops.function.math_func import addmm_ext as addmm

# 505
from mindspore.ops.function.math_func import addmv_ext as addmv

# 510
from mindspore.ops.function.math_func import amax_ext as amax

# 511
from mindspore.ops.function.math_func import amin_ext as amin

# 521
from mindspore.ops.functional_overload import bitwise_not

# 526
from mindspore.ops.auto_generate import dot

# 533
from mindspore.ops.function.math_func import frac_ext as frac

# 538
from mindspore.ops.function.math_func import histc_ext as histc

# 552
from mindspore.ops.auto_generate import log10_ext as log10

# 553
from mindspore.ops.auto_generate import logaddexp_ext as logaddexp

# 557
from mindspore.ops.auto_generate import logsumexp_ext as logsumexp

# 582
from mindspore.ops.function.math_func import std_mean_ext as std_mean

# 588
from mindspore.ops.function.math_func import var_mean_ext as var_mean

# 610
from mindspore.ops.function.math_func import nan_to_num

# 613
from mindspore.ops.functional_overload import nansum

# 664
from mindspore.ops.function.array_func import meshgrid_ext as meshgrid

# 695
from mindspore.ops.auto_generate import count_nonzero

# 697
from mindspore.ops.function.math_func import float_power_ext as float_power

# 708
from mindspore.ops.function.math_func import std_ext as std

# 887
from mindspore.ops.auto_generate import log2_ext as log2

# 889
from mindspore.ops.function.math_func import isnan_ext as isnan

# 1007
from mindspore.ops.auto_generate import t_ext as t
from mindspore.ops.auto_generate.pyboost_inner_prim import squeeze_impl
from mindspore.ops.auto_generate.gen_ops_prim import equal_ext_op


# 1023
from mindspore.ops.function.array_func import unbind_ext as unbind


def add(input, other, *, alpha=1):
    r"""
    Adds scaled other value to input Tensor.

    .. math::

        out_{i} = input_{i} + alpha \times other_{i}

    Note:
        - When the two inputs have different shapes,
          they must be able to broadcast to a common shape.
        - The two inputs and alpha comply with the implicit type conversion rules to make the data types
          consistent.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_.
        other (Union[Tensor, number.Number, bool]): The second input, is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_.

    Keyword Args:
        alpha (number.Number): A scaling factor applied to `other`, default: ``1``.

    Returns:
        Tensor with a shape that is the same as the broadcasted shape of the input `input` and `other`,
        and the data type is the one with higher precision or higher digits among the two inputs and alpha.

    Raises:
        TypeError: If the type of `input`, `other`, or `alpha` is not one of the following: Tensor, number.Number, bool.
        TypeError: If `alpha` is of type float but `input` and `other` are not of type float.
        TypeError: If `alpha` is of type bool but `input` and `other` are not of type bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import mint
        >>> x = Tensor(1, mindspore.int32)
        >>> y = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> alpha = 0.5
        >>> output = mint.add(x, y, alpha=alpha)
        >>> print(output)
        [3. 3.5 4.]
        >>> # the data type of x is int32, the data type of y is float32,
        >>> # alpha is a float, and the output is the data format of higher precision float32.
        >>> print(output.dtype)
        Float32
    """
    return ops.auto_generate.add_ext(input, other, alpha)


def any(input, dim=None, keepdim=False):
    r"""
    Reduces a dimension of `input` by the "logical OR" of all elements in the dimension, by default. And also can
    reduce a dimension of `input` along the `dim`. Determine whether the dimensions of the output and input are the
    same by controlling `keepdim`.

    Note:
        The `dim` with tensor type is only used for compatibility with older versions and is not recommended.

    Args:
        input (Tensor): Input Tensor, has the shape :math:`(N, *)` where :math:`*` means,
            any number of additional dimensions.
        dim (Union[int, tuple(int), list(int), Tensor], optional): The dimensions to reduce.
            Suppose the rank of `input` is r, `dim` must be in the range [-r,r).
            Default: ``None`` , all dimensions are reduced.
        keepdim (bool, optional): If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , don't keep these dimensions. Default : ``False`` .

    Returns:
        Tensor, the dtype is bool.

        - If `dim` is ``None`` , and `keepdim` is ``False`` ,
          the output is a 0-D Tensor representing the "logical OR" of all elements in the input Tensor.
        - If `dim` is int, such as 2, and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_1, input_3, ..., input_R)`.
        - If `dim` is tuple(int) or list(int), such as (2, 3), and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_1, input_4, ..., input_R)`.
        - If `dim` is 1-D Tensor, such as [2, 3], and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_1, input_4, ..., input_R)`.

    Raises:
        TypeError: If `keepdim` is not a bool.
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not one of the following: int, tuple, list or Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[True, False], [True, True]]))
        >>> # case 1: Reduces a dimension by the "logical OR" of all elements in the dimension.
        >>> output = mint.any(x, keepdim=True)
        >>> print(output)
        [[ True]]
        >>> print(output.shape)
        (1, 1)
        >>> # case 2: Reduces a dimension along dim 0.
        >>> output = mint.any(x, dim=0)
        >>> print(output)
        [ True True]
        >>> # case 3: Reduces a dimension along dim 1.
        >>> output = mint.any(x, dim=1)
        >>> print(output)
        [ True True]
    """
    return ops.functional.any(input, dim, keepdim)


def all(input, dim=None, keepdim=False):
    r"""
    all(input) -> Tensor

    Reduces all elements of `input` by the "logical AND".

    Args:
        input (Tensor): Input Tensor, has the shape :math:`(N, *)` where :math:`*` means,
            any number of additional dimensions.

    Returns:
        Tensor, the dtype is bool.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[True, False], [True, True]]))
        >>> # case 1: Reduces a dimension by the "logicalAND" of all elements in the dimension.
        >>> output = mint.all(x)
        >>> print(output)
        False

    .. function:: all(input, dim, keepdim=False) -> Tensor
        :noindex:

    Reduces a dimension of `input` by the "logical AND" of all elements in the dimension, by default. And also can
    reduce a dimension of `input` along the `dim`. Determine whether the dimensions of the output and input are the
    same by controlling `keepdim`.

    Note:
        The `dim` with tensor type is only used for compatibility with older versions and is not recommended.

    Args:
        input (Tensor): Input Tensor, has the shape :math:`(N, *)` where :math:`*` means,
            any number of additional dimensions.
        dim (Union[int, tuple(int), list(int), Tensor]): The dimensions to reduce.
            Suppose the rank of `input` is r, `dim` must be in the range [-rank(input), rank(input)).
        keepdim (bool, optional): If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , don't keep these dimensions. Default : ``False`` .

    Returns:
        Tensor, the dtype is bool.

        - If `dim` is int, such as 2, and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_1, input_3, ..., input_R)`.
        - If `dim` is tuple(int) or list(int), such as (2, 3), and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_1, input_4, ..., input_R)`.
        - If `dim` is 1-D Tensor, such as [2, 3], and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_1, input_4, ..., input_R)`.

    Raises:
        TypeError: If `keepdim` is not a bool.
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not one of the following: int, tuple, list or Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[True, False], [True, True]]))
        >>> # case 1: Reduces a dimension along axis 0.
        >>> output = mint.all(x, dim=0)
        >>> print(output)
        [ True False]
        >>> # case 2: Reduces a dimension along axis 1.
        >>> output = mint.all(x, dim=1)
        >>> print(output)
        [False True]
    """
    return ops.function.math_func.all(input, dim, keepdim)


def allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False):
    """
    Returns a new Tensor with boolean elements representing if each element of `input`
    is “close” to the corresponding element of `other`. Closeness is defined as:

    .. math::
        |input-other| ≤ atol + rtol × |other|

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): First tensor to compare.
            Support dtype: float16, float32, float64, int8, int16, int32, int64 and uint8.
            On Ascend, more dtypes are support: bool and bfloat16.
        other (Tensor): Second tensor to compare. Dtype must be same as `input`.
        rtol (Union[float, int, bool], optional): Relative tolerance. Default: ``1e-05`` .
        atol (Union[float, int, bool], optional): Absolute tolerance. Default: ``1e-08`` .
        equal_nan (bool, optional): If ``True`` , then two NaNs will be considered equal. Default: ``False``.

    Returns:
        A bool Scalar.

    Raises:
        TypeError: `input` or `other` is not Tensor.
        TypeError: `input` or `other` dtype is not support.
        TypeError: `atol` or `rtol` is not float, int or bool.
        TypeError: `equal_nan` is not bool.
        TypeError: `input` and `other` have different dtypes.
        ValueError: `input` and `other` cannot broadcast.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> input = Tensor(np.array([1.3, 2.1, 3.2, 4.1, 5.1]), mindspore.float16)
        >>> other = Tensor(np.array([1.3, 3.3, 2.3, 3.1, 5.1]), mindspore.float16)
        >>> output = mint.allclose(input, other)
        >>> print(output)
        False
    """
    return isclose(input, other, rtol, atol, equal_nan).all().item()


def cat(tensors, dim=0):
    r"""
    Connect input tensors along with the given dimension.

    The input data is a tuple or a list of tensors. These tensors have the same rank :math:`R`.
    Set the given dimension as :math:`m`, and :math:`0 \le m < R`. Set the number of input tensors as :math:`N`.
    For the :math:`i`-th tensor :math:`t_i`, it has the shape of :math:`(x_1, x_2, ..., x_{mi}, ..., x_R)`.
    :math:`x_{mi}` is the :math:`m`-th dimension of the :math:`t_i`. Then, the shape of the output tensor is

    .. math::

        (x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)

    Args:
        tensors (Union[tuple, list]): A tuple or a list of input tensors.
            Suppose there are two tensors in this tuple or list, namely t1 and t2.
            To perform `concat` in the dimension 0 direction, except for the :math:`0`-th dimension,
            all other dimensions should be equal, that is,
            :math:`t1.shape[1] = t2.shape[1], t1.shape[2] = t2.shape[2], ..., t1.shape[R-1] = t2.shape[R-1]`,
            where :math:`R` represents the rank of tensor.
        dim (int, optional): The specified dimension, whose value is in range :math:`[-R, R)`. Default: ``0`` .

    Returns:
        Tensor, the shape is :math:`(x_1, x_2, ..., \sum_{i=1}^Nx_{mi}, ..., x_R)`.
        The data type is the same with `tensors`.

    Raises:
        TypeError: If `dim` is not an int.
        ValueError: If `tensors` have different dimension of tensor.
        ValueError: If `dim` not in range :math:`[-R, R)`.
        ValueError: If tensor's shape in `tensors` except for `dim` are different.
        ValueError: If `tensors` is an empty tuple or list.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import mint
        >>> input_x1 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> input_x2 = Tensor(np.array([[0, 1], [2, 1]]).astype(np.float32))
        >>> output = mint.cat((input_x1, input_x2))
        >>> print(output)
        [[0. 1.]
         [2. 1.]
         [0. 1.]
         [2. 1.]]
        >>> output = mint.cat((input_x1, input_x2), 1)
        >>> print(output)
        [[0. 1. 0. 1.]
         [2. 1. 2. 1.]]
    """
    return ops.auto_generate.cat(tensors, dim)


def concat(tensors, dim=0):
    r"""
    Alias for :func:`mindspore.mint.cat`.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    return cat(tensors, dim)


def cummax(input, dim):
    r"""
    Returns a tuple (values, indices) where `values` is the cumulative maximum value of input Tensor `input`
    along the dimension `dim`, and `indices` is the index location of each maximum value.

    .. math::
        \begin{array}{ll} \\
            y_{i} = \max(x_{1}, x_{2}, ... , x_{i})
        \end{array}

    .. note::
        O2 mode is not supported in Ascend.

    Args:
        input (Tensor): The input Tensor. Rank of `input` must be greater than 0.
        dim (int): The dimension to do the operation over. The value of `dim` must be in the range
            `[-input.ndim, input.ndim - 1]`.

    Returns:
        tuple [Tensor], tuple of 2 Tensors, containing the cumulative maximum of elements and the index.
        The shape of each output tensor is the same as that of input `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not an int.
        ValueError: If `dim` is out the range of `[-input.ndim, input.ndim - 1]`.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> from mindspore import ops
        >>> x = Tensor(np.array([[3, 4, 6, 10], [1, 6, 7, 9], [4, 3, 8, 7], [1, 3, 7, 9]]).astype(np.float32))
        >>> output = mint.cummax(x, dim=0)
        >>> print(output[0])
        [[ 3.  4.  6. 10.]
         [ 3.  6.  7. 10.]
         [ 4.  6.  8. 10.]
         [ 4.  6.  8. 10.]]
        >>> print(output[1])
        [[0 0 0 0]
         [0 1 1 0]
         [2 1 2 0]
         [2 1 2 0]]
    """
    return ops.auto_generate.cummax(input, dim)


def not_equal(input, other):
    r"""
    Alias for :func:`mindspore.mint.ne` .

    Supported Platforms:
        ``Ascend``
    """
    return ne(input, other)


def softmax(input, dim, *, dtype=None):
    r"""
    Alias for :func:`mindspore.mint.nn.functional.softmax`.

    Supported Platforms:
        ``Ascend``
    """
    return softmax_ext(input, dim, dtype)


def _einsum_convert_sublist_to_label(num, ell_num=False):
    """Convert sublist to label."""
    if num == Ellipsis or ell_num and num == 52:
        return '...'
    if 0 <= num < 26:
        return chr(num + ord('A'))
    if 26 <= num < 52:
        return chr(num + ord('a') - 26)
    raise ValueError(
        f'For einsum, the number in sublist must be in range [0, 52), but got {num}')


def _einsum_convert_label_to_index(label):
    """Convert label to index."""
    label_num = ord(label)
    if ord('A') <= label_num <= ord('Z'):
        return label_num - ord('A')
    if ord('a') <= label_num <= ord('z'):
        return label_num - ord('a') + 26
    if label_num == ord('.'):
        return 52
    raise ValueError(
        f'For einsum, the label in equation must be in [a-zA-Z] or ., but got {label}')


def _einsum_convert_sublist(equation, *operands):
    """Convert the sublist to an equation operand if the received input is a sublist format."""
    if isinstance(equation, Tensor):
        equation_tmp = ''
        for i, lst in enumerate(operands):
            if i % 2 == 0:
                for _, num in enumerate(lst):
                    equation_tmp += _einsum_convert_sublist_to_label(num)
                if i in (len(operands) - 1, len(operands) - 2):
                    continue
                equation_tmp += ','
        if len(operands) % 2 == 0:
            equation_tmp += '->'
            for _, num in enumerate(operands[-1]):
                equation_tmp += _einsum_convert_sublist_to_label(num)
            operands_tmp = list([equation]) + list(operands[1:-1:2])
        else:
            operands_tmp = list([equation]) + list(operands[1::2])
        equation = equation_tmp
        operands = tuple(operands_tmp)
    if len(operands) == 0:  # pylint: disable=len-as-condition
        raise ValueError(
            "For einsum, the 'operands' must have at least one operand.")
    return equation, operands


def _einsum_check_inputargs(equation, operands):
    """Check equation and operands."""
    if not isinstance(equation, str):
        raise TypeError(
            f"For einsum, 'equation' must be a str, but got {type(equation)}.")
    for operand in operands:
        if not isinstance(operand, Tensor):
            raise TypeError(
                f"For einsum, members of 'operands' must be Tensor, but got {type(operand)}.")


@constexpr
def _einsum_parse_equation(equation):
    """Parse equation."""
    l_equation = ''
    r_equation = ''
    equation = equation.replace(' ', '')

    if '->' in equation:
        l_equation, r_equation = equation.split('->', 1)
        if l_equation == '':
            raise ValueError(
                'For einsum, equation must contain characters to the left fo the arrow.')
    else:
        l_equation = equation

    if ',' in l_equation:
        l_equationlst = l_equation.split(",")
    else:
        l_equationlst = [l_equation]

    l_equationlst = []

    for subequation in l_equation.split(','):
        if '.' in subequation and ('...' not in subequation or subequation.count('.') != 3):
            raise ValueError(f"For einsum, an ellipsis in the equation must include three continuous \'.\', "
                             f"and can only be found once.")
        subequation_lst = [_einsum_convert_label_to_index(
            label) for label in subequation.replace('...', '.')]
        l_equationlst.append(subequation_lst)

    if "." in r_equation and ('...' not in r_equation or r_equation.count('.') != 3):
        raise ValueError(f"For einsum, an ellipsis in the equation must include three continuous \'.\', "
                         f"and can only be found once.")
    r_equationlst = [_einsum_convert_label_to_index(
        label) for label in r_equation.replace('...', '.')]

    return l_equationlst, r_equationlst, ('->' in equation)


def _einsum_parse_labels(l_equationlst, operands):
    """Parse left script of equation."""
    align_rank = 0
    max_labels = 53
    ellipsis_dimnum = 0
    labels_count = [0] * max_labels

    if len(operands) != len(l_equationlst):
        raise ValueError(f"For einsum, 'operands' is not equal to specified in the 'equation', "
                         f"but got {len(operands)} and {len(l_equationlst)}.")

    for idx, sub_equ in enumerate(l_equationlst):
        start_dim = 0
        label_num = 0
        operand_shape = list(operands[idx].shape)
        for label in sub_equ:
            dim_num = 1
            label_num += 1
            end_dim = start_dim + 1

            # Label is ellipsis
            if label == 52:
                end_dim = len(operand_shape) - len(sub_equ) + label_num
                dim_num = end_dim - start_dim
                if ellipsis_dimnum != 0 and ellipsis_dimnum != dim_num:
                    raise ValueError(f"For einsum, an ellipsis in 'equation' can only represent the same numbers of "
                                     f"dimensions in 'operands'.")
                ellipsis_dimnum = dim_num
            if labels_count[label] == 0:
                align_rank += dim_num
            labels_count[label] += 1
            start_dim += dim_num
        if label_num != len(sub_equ) or start_dim != len(operand_shape):
            raise ValueError(f"For einsum, the numbers of labels specified in the 'equation' does not match "
                             f"'operands[{idx}]'.")
    return ellipsis_dimnum, labels_count, align_rank


def _einsum_infer_output(r_equationlst, arrow_exist, ellipsis_dimnum, labels_count):
    """Parse right script of equation and infer output shape."""
    idx = 0
    idle_idx = -1
    output_rank = 0
    labels_perm_idx = [idle_idx] * 53

    if arrow_exist:
        for label in r_equationlst:
            if labels_count[label] != 0:
                if labels_perm_idx[label] != idle_idx:
                    raise ValueError(f"For einsum, '{_einsum_convert_sublist_to_label(label, True)}' or {label} in "
                                     f"sublist format has appears more than once in output subscript.")
                dimnum = 1
                if label == 52:
                    dimnum = ellipsis_dimnum
                labels_perm_idx[label] = idx
                output_rank += dimnum
                idx += dimnum
            else:
                raise ValueError(f"For einsum, the label to the right of arrow in the 'equation' must appear on "
                                 f"left, but '{_einsum_convert_sublist_to_label(label, True)}' does not.")
    else:
        if labels_count[52] != 0:
            output_rank += ellipsis_dimnum
            labels_perm_idx[52] = idx
            idx += ellipsis_dimnum
        for label, count in enumerate(labels_count):
            if count == 1:
                output_rank += 1
                labels_perm_idx[label] = idx
                idx += 1

    for label, count in enumerate(labels_count):
        if count != 0 and labels_perm_idx[label] == idle_idx:
            labels_perm_idx[label] = idx
            idx += 1

    return output_rank, labels_perm_idx


def _einsum_adjust_operands(operands, l_equationlst, ellipsis_dimnum, labels_perm_idx, align_rank):
    """Align operands to output as possible."""
    # Unsqueeze miss dimensions to make all operands has same rank, compute diagonal if operand has same label.
    # Then use _labels_perm_idx to transpose all operands to align dimensions with output.
    adjust_operands = []
    for idx, operand in enumerate(operands):
        idle_dim = -1
        align_axis = [idle_dim] * align_rank
        label_dims = [idle_dim] * 53
        dim = 0

        for label in l_equationlst[idx]:
            if label_dims[label] != idle_dim:
                operand = ops.diagonal(operand, 0, label_dims[label], dim)
                diag_perm = []
                diag_dim = 0
                for i in range(len(operand.shape)):
                    if i == label_dims[label]:
                        diag_perm.append(len(operand.shape) - 1)
                    else:
                        diag_perm.append(diag_dim)
                        diag_dim += 1
                operand = permute(operand, tuple(diag_perm))
            else:
                label_dims[label] = dim
                if label == 52:
                    for ell_idx in range(ellipsis_dimnum):
                        align_axis[labels_perm_idx[label] + ell_idx] = dim
                        dim += 1
                else:
                    align_axis[labels_perm_idx[label]] = dim
                    dim += 1
        if len(operand.shape) < align_rank:
            for i, axis in enumerate(align_axis):
                if axis == idle_dim:
                    align_axis[i] = dim
                    dim += 1
            missing_dims = [1] * (align_rank - len(operand.shape))
            operand_shape = list(operand.shape) + missing_dims
            operand = reshape(operand, operand_shape)
        operand = permute(operand, tuple(align_axis))
        adjust_operands.append(operand)
    return adjust_operands


def _einsum_find_dimlastop(align_rank, operands, adjust_operands):
    """Find dim last operand."""
    dim_last_op = [0] * align_rank
    has_zero_dim = False
    for dim in range(align_rank):
        broadcast_dim = adjust_operands[0].shape[dim]
        for idx in range(1, len(adjust_operands)):
            other_dim = adjust_operands[idx].shape[dim]
            if broadcast_dim != other_dim and broadcast_dim != 1 and other_dim != 1:
                err_msg = "For einsum, operands do not broadcast after align to output [shapes :origin -> adjust]:"
                for i in range(len(operands)):
                    err_msg += f" {operands[i].shape} -> {adjust_operands[i].shape}"
                raise ValueError(err_msg)
            if other_dim != 1:
                dim_last_op[dim] = idx
                broadcast_dim = other_dim
        has_zero_dim = has_zero_dim or broadcast_dim == 0
    return dim_last_op, has_zero_dim


def _einsum_multiplication(sum_dims, l_tensor, r_tensor):
    """Compute bmm for einsum."""
    batch_dims = []
    lonly_dims = []
    ronly_dims = []
    batch_size = 1
    lonly_size = 1
    ronly_size = 1
    sum_size = 1

    l_shape = l_tensor.shape
    r_shape = r_tensor.shape

    # Compute sum if dim is in sum_dims and get shapes for bmm
    for i in range(len(l_shape)):
        sum_l = l_shape[i] > 1
        sum_r = r_shape[i] > 1
        if i in sum_dims:
            if sum_l and sum_r:
                sum_size *= l_shape[i]
            elif sum_l:
                l_tensor = sum(l_tensor, i, True)
            elif sum_r:
                r_tensor = sum(r_tensor, i, True)
        elif sum_l and sum_r:
            batch_dims.append(i)
            batch_size *= l_shape[i]
        elif sum_l:
            lonly_dims.append(i)
            lonly_size *= l_shape[i]
        else:
            ronly_dims.append(i)
            ronly_size *= r_shape[i]

    # Compute the einsum bmm operators pipeline.
    # The whole operators pipeline is transpose(in) -> reshape(in) -> bmm(in) -> reshape(out) -> transpose(out).
    l_reshape_shape = (batch_size, lonly_size, sum_size)
    r_reshape_shape = (batch_size, sum_size, ronly_size)

    out_reshape_shape = [l_shape[dim] for dim in batch_dims]
    out_reshape_shape += [l_shape[dim] for dim in lonly_dims]
    out_reshape_shape += [1 for _ in sum_dims]
    out_reshape_shape += [r_shape[dim] for dim in ronly_dims]

    l_perm_axis = batch_dims + lonly_dims + sum_dims + ronly_dims
    r_perm_axis = batch_dims + sum_dims + ronly_dims + lonly_dims
    out_perm_axis = [-1] * len(out_reshape_shape)

    out_dim = 0
    for idx in range(len(l_perm_axis)):
        out_perm_axis[l_perm_axis[idx]] = out_dim
        out_dim += 1

    l_tensor = permute(l_tensor, tuple(l_perm_axis))
    l_tensor = reshape(l_tensor, l_reshape_shape)

    r_tensor = permute(r_tensor, tuple(r_perm_axis))
    r_tensor = reshape(r_tensor, r_reshape_shape)

    output = bmm(l_tensor, r_tensor)
    output = reshape(output, out_reshape_shape)
    output = permute(output, tuple(out_perm_axis))

    output_origin_shape = output.shape
    output_squeeze_shape = []
    for dim in range(len(output_origin_shape)):
        if dim not in sum_dims:
            output_squeeze_shape.append(output_origin_shape[dim])

    return reshape(output, output_squeeze_shape)


def _einsum(equation, operands):
    '''Einsum main process'''
    _l_equationlst, _r_equationlst, _arrow_exist = _einsum_parse_equation(
        equation)
    _ellipsis_dimnum, _labels_count, _align_rank = _einsum_parse_labels(
        _l_equationlst, operands)
    _output_rank, _labels_perm_idx = _einsum_infer_output(
        _r_equationlst, _arrow_exist, _ellipsis_dimnum, _labels_count)
    _adjust_operands = _einsum_adjust_operands(operands, _l_equationlst, _ellipsis_dimnum, _labels_perm_idx,
                                               _align_rank)
    _dim_last_op, _has_zero_dim = _einsum_find_dimlastop(
        _align_rank, operands, _adjust_operands)
    _result = _adjust_operands[0]

    # Fast path if operands has zero dim.
    if _has_zero_dim:
        output_shape = []
        for dim in range(_output_rank):
            output_shape.append(_adjust_operands[_dim_last_op[dim]].shape[dim])
        return zeros(output_shape, dtype=_result.dtype)

    # Sum or squeeze dimensions that is 1 for all rest operands.
    _reduce_dim = _output_rank
    for dim in range(_output_rank, _align_rank):
        if _dim_last_op[dim] == 0:
            if _result.shape[_reduce_dim] == 1:
                _result = squeeze(_result, _reduce_dim)
            else:
                _result = sum(_result, _reduce_dim)
        else:
            _reduce_dim += 1

    # Compute multiplication if operands are more than two.
    for i in range(1, len(_adjust_operands)):
        operand = _adjust_operands[i]
        dim = _output_rank
        sum_dims = []
        for j in range(_output_rank, _align_rank):
            if _dim_last_op[j] < i:
                operand = squeeze(operand, dim)
            elif _dim_last_op[j] == i:
                if _result.shape[dim] == 1:
                    operand = sum(operand, dim)
                    _result = squeeze(_result, dim)
                else:
                    sum_dims.append(dim)
                    dim += 1
            else:
                dim += 1

        if sum_dims == []:
            _result = mul(_result, operand)
        elif len(sum_dims) == len(_result.shape):
            _result = ops.auto_generate.dot(flatten(_result), flatten(operand))
        else:
            _result = _einsum_multiplication(sum_dims, _result, operand)

    return _result


def einsum(equation, *operands):
    r"""
    According to the Einstein summation Convention (Einsum),
    the product of the input tensor elements is summed along the specified dimension.
    You can use this operator to perform diagonal, reducesum, transpose, matmul, mul, inner product operations, etc.

    Note:
        The sublist format is also supported. For example, mint.einsum(op1, sublist1, op2, sublist2, ..., sublist_out).
        In this format, equation can be derived by the sublists which are made up of Python's Ellipsis and list of
        integers in [0, 52). Each operand is followed by a sublist and an output sublist is at the end.
        Dynamic shape, dynamic rank input is not supported in `graph mode (mode=mindspore.GRAPH_MODE)
        <https://www.mindspore.cn/docs/en/master/model_train/program_form/static_graph.html>`_.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        equation (str): Notation based on the Einstein summation convention, represent the operation you want to do.
            the value can contain only letters, commas, ellipsis and arrow. The letters(must be in [a-zA-Z]) represent
            input tensor dimension, commas(,) represent separate tensors, ellipsis indicates the tensor dimension that
            you do not care about, the left of the arrow indicates the input tensors, and the right of it indicates the
            desired output dimension. If there are no arrows in the equation, the letters that appear exactly once in
            the equation will be part of the output, sorted in increasing alphabetical order. The output is computed by
            multiplying the input operands element-wise, with their dimensions aligned based on the letters, and then
            summing out the dimensions whose letters are not part of the output. If there is one arrow in the equation,
            the output letters must appear at least once for some input operand and at most once for the output.
        operands (Tensor): Input tensor used for calculation. The dtype of the tensor must be the same.

    Returns:
        Tensor, the shape of it can be obtained from the `equation` , and the dtype is the same as input tensors.

    Raises:
        TypeError: If `equation` is invalid, or the `equation` does not match the input tensor.
        ValueError: If the number in sublist is not in [0, 52) in sublist format.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> equation = "i->"
        >>> output = mint.einsum(equation, x)
        >>> print(output)
        [7.]
        >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
        >>> equation = "i,i->i"
        >>> output = mint.einsum(equation, x, y)
        >>> print(output)
        [ 2. 8. 12.]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> y = Tensor(np.array([[2.0, 3.0], [1.0, 2.0], [4.0, 5.0]]), mindspore.float32)
        >>> equation = "ij,jk->ik"
        >>> output = mint.einsum(equation, x, y)
        >>> print(output)
        [[16. 22.]
         [37. 52.]]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->ji"
        >>> output = mint.einsum(equation, x)
        >>> print(output)
        [[1. 4.]
         [2. 5.]
         [3. 6.]]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "ij->j"
        >>> output = mint.einsum(equation, x)
        >>> print(output)
        [5. 7. 9.]
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), mindspore.float32)
        >>> equation = "...->"
        >>> output = mint.einsum(equation, x)
        >>> print(output)
        [21.]
        >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
        >>> y = Tensor(np.array([2.0, 4.0, 1.0]), mindspore.float32)
        >>> equation = "j,i->ji"
        >>> output = mint.einsum(equation, x, y)
        >>> print(output)
        [[ 2. 4. 1.]
         [ 4. 8. 2.]
         [ 6. 12. 3.]]
        >>> x = mindspore.Tensor([1, 2, 3, 4], mindspore.float32)
        >>> y = mindspore.Tensor([1, 2], mindspore.float32)
        >>> output = mint.einsum(x, [..., 1], y, [..., 2], [..., 1, 2])
        >>> print(output)
        [[1. 2.]
         [2. 4.]
         [3. 6.]
         [4. 8.]]
    """
    _equation, _operands = _einsum_convert_sublist(equation, *operands)
    _einsum_check_inputargs(_equation, _operands)

    for operand in _operands:
        if ops.is_sequence_shape_unknown(operand.shape) or ops.is_sequence_value_unknown(operand.shape):
            raise ValueError(f"For einsum, the element of 'operands' can't be dynamic shape or dynamic rank.")

    return _einsum(_equation, _operands)


def equal(input, other):
    r"""
    Computes the equivalence between two tensors.

    Note:
        `input` and `other` comply with the implicit type conversion rules to make the data types consistent.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The first input.
        other (Tensor): The second input.

    Returns:
        bool.

    Raises:
        TypeError: If `input` or `other` is not a Tensor.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore
        >>> from mindspore import Tensor, mint
        >>> x = Tensor([1, 2, 3], mindspore.int32)
        >>> y = Tensor([1, 2, 4], mindspore.int32)
        >>> output = mint.equal(x, y)
        >>> print(output)
        False
    """
    result = equal_ext_op(input, other)
    return result.item()


def isfinite(input):
    r"""
    Determine which elements are finite for each position. If elements are not ``NaN`` , ``-INF`` , ``INF``,
    they are finite.

    .. math::
        out_i = \begin{cases}
          & \text{ if } input_{i} = \text{Finite},\ \ True \\
          & \text{ if } input_{i} \ne \text{Finite},\ \ False
        \end{cases}

    Args:
        input (Tensor): The input tensor.

    Returns:
        Tensor, has the same shape of input, and the dtype is bool.

    Raises:
        TypeError: If input is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([np.log(-1), 1, np.log(0)]), mindspore.float32)
        >>> output = mint.isfinite(x)
        >>> print(output)
        [False True False]
        >>> x = Tensor(2.1, mindspore.float64)
        >>> output = mint.isfinite(x)
        >>> print(output)
        True
    """
    return ops.auto_generate.isfinite(input)


def item(input):
    r"""
    Returns the value of this tensor as a standard Python number.

    Note:
        This only works for tensors with one element.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.

    Returns:
        number.

    Raises:
        TypeError: If `input` is not a Tensor.
        RuntimeError: If the number of `input` elements is not 1.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([1]).astype(np.float32))
        >>> result = mint.item(x)
        >>> print(result)
        1.0
    """
    if not isinstance(input, Tensor):
        raise TypeError(f"the input must be a Tensor, but got {type(input)}")
    if input.size != 1:
        raise RuntimeError(
            "a Tensor with {} elements cannot be converted to Scalar".format(input.size))
    return input.asnumpy().item()


def mean(input, dim=None, keepdim=False, *, dtype=None):
    r"""
    mean(input, *, dtype=None) -> Tensor

    Reduces all dimension of a tensor by averaging all elements.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        Tensor.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        ... [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ... [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]),
        ... mindspore.float32)
        >>> output = mint.mean(x)
        >>> print(output)
        5.0
        >>> print(output.shape)
        ()

    .. function:: mean(input, dim, keepdim=False, *, dtype=None) -> Tensor
        :noindex:

    Reduces all dimension of a tensor by averaging all elements in the dimension, by default.
    And reduce a dimension of `input` along the specified `dim`. `keepdim`
    determines whether the dimensions of the output and input are the same.

    Note:
        The `dim` with tensor type is only used for compatibility with older versions and is not recommended.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        dim (Union[int, tuple(int), list(int), Tensor]): The dimensions to reduce.
            Only constant value is allowed. Assume the rank of `input` is r, and the value range is [-r,r).
        keepdim (bool): If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , don't keep these dimensions. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        Tensor.

        - If `dim` is int, set as 1, and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_0, input_2, ..., input_R)`.
        - If `dim` is tuple(int) or list(int), set as (1, 2), and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_0, input_3, ..., input_R)`.
        - If `dim` is 1-D Tensor, set as [1, 2], and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_0, input_3, ..., input_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not one of the following: int, tuple, list or Tensor.
        TypeError: If `keepdim` is not a bool.
        ValueError: If `dim` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[[2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2]],
        ... [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ... [[6, 6, 6, 6, 6, 6], [8, 8, 8, 8, 8, 8], [10, 10, 10, 10, 10, 10]]]),
        ... mindspore.float32)
        >>> output = mint.mean(x, 0, True)
        >>> print(output)
        [[[4. 4. 4. 4. 4. 4.]
          [5. 5. 5. 5. 5. 5.]
          [6. 6. 6. 6. 6. 6.]]]
    """
    return ops.auto_generate.mean_ext(input, dim, keepdim, dtype)


def prod(input, dim=None, keepdim=False, *, dtype=None):
    r"""
    prod(input, *, dtype=None) -> Tensor

    Multiplying all elements of input.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        Tensor.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = mint.prod(x)
        >>> print(output)
        2.2833798e+33
        >>> print(output.shape)
        ()

    .. function:: prod(input, dim, keepdim=False, *, dtype=None) -> Tensor
        :noindex:

    Reduces a dimension of a tensor by multiplying all elements in the dimension, by default. And also can
    reduce a dimension of `input` along the `dim`. Determine whether the dimensions of the output and input are the
    same by controlling `keepdim`.

    Args:
        input (Tensor[Number]): The input tensor. The dtype of the tensor to be reduced is number.
            :math:`(N, *)` where :math:`*` means, any number of additional dimensions.
        dim (int): The dimensions to reduce. Only constant value is allowed.
            Assume the rank of `x` is r, and the value range is [-r,r).
        keepdim (bool): If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , don't keep these dimensions. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        Tensor.

        - If `dim` is int, set as 1, and `keepdim` is ``False`` ,
          the shape of output is :math:`(input_0, input_2, ..., input_R)`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not int.
        TypeError: If `keepdim` is not a bool.
        ValueError: If `dim` is out of range.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mindspore.float32)
        >>> output = mint.prod(x, 0, True)
        >>> print(output)
        [[[ 28.  28.  28.  28.  28.  28.]
          [ 80.  80.  80.  80.  80.  80.]
          [162. 162. 162. 162. 162. 162.]]]
    """
    return ops.auto_generate.prod_ext(input, dim, keepdim, dtype)


def sum(input, dim=None, keepdim=False, *, dtype=None):
    r'''
    sum(input, *, dtype=None) -> Tensor

    Calculate sum of all elements in Tensor.

    Args:
        input (Tensor): The input tensor.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        A Tensor, sum of all elements in `input`.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mstype.float32)
        >>> out = mint.sum(x)
        >>> print(out)
        270.0

    .. function:: sum(input, dim, keepdim=False, *, dtype=None) -> Tensor
        :noindex:

    Calculate sum of Tensor elements over a given dim.

    Note:
        The `dim` with tensor type is only used for compatibility with older versions and is not recommended.

    Args:
        input (Tensor): The input tensor.
        dim (Union[int, tuple(int), list(int), Tensor]): Dimensions along which a sum is performed.
            If the `dim` is a tuple or list of ints, a sum is performed on all the dimensions specified in the tuple.
            Must be in the range :math:`[-input.ndim, input.ndim)` .
        keepdim (bool): Whether the output tensor has `dim` retained or not.
            If ``True`` , keep these reduced dimensions and the length is 1.
            If ``False`` , don't keep these dimensions. Default: ``False`` .

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The desired data type of returned Tensor. Default: ``None`` .

    Returns:
        A Tensor, sum of elements over a given `dim` in `input`.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If `dim` is not an int, tulpe(int), list(int) or Tensor.
        ValueError: If `dim` is not in the range :math:`[-input.ndim, input.ndim)` .
        TypeError: If `keepdim` is not a bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(np.array([[[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2], [3, 3, 3, 3, 3, 3]],
        ...                      [[4, 4, 4, 4, 4, 4], [5, 5, 5, 5, 5, 5], [6, 6, 6, 6, 6, 6]],
        ...                      [[7, 7, 7, 7, 7, 7], [8, 8, 8, 8, 8, 8], [9, 9, 9, 9, 9, 9]]]), mstype.float32)
        >>> out = mint.sum(x)
        >>> print(out)
        270.0
        >>> out = mint.sum(x, dim=2)
        >>> print(out)
        [[ 6. 12. 18.]
         [24. 30. 36.]
         [42. 48. 54.]]
        >>> out = mint.sum(x, dim=2, keepdim=True)
        >>> print(out)
        [[[ 6.]
          [12.]
          [18.]]
         [[24.]
          [30.]
          [36.]]
         [[42.]
          [48.]
          [54.]]]
    '''
    return ops.auto_generate.sum_ext(input, dim, keepdim, dtype)


def ones(size, *, dtype=None):
    r"""
    Creates a tensor filled with value ones.

    Creates a tensor with shape described by the first argument and fills it with value ones in type of the second
    argument.

    Args:
        size (Union[tuple[int], list[int], int, Tensor]): The specified shape of output tensor. Only positive integer or
            tuple or Tensor containing positive integers are allowed. If it is a Tensor,
            it must be a 0-D or 1-D Tensor with int32 or int64 dtypes.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The specified type of output tensor. If `dtype` is ``None`` ,
            `mindspore.float32` will be used. Default: ``None`` .

    Returns:
        Tensor, whose dtype and size are defined by input.

    Raises:
        TypeError: If `size` is neither an int nor a tuple/list/Tensor of int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import mint
        >>> output = mint.ones((2, 2), dtype=mindspore.float32)
        >>> print(output)
        [[1. 1.]
         [1. 1.]]
    """
    return ops.auto_generate.ones(size, dtype)


def permute(input, dims):
    """
    Permutes the dimensions of the input tensor according to input `dims` .

    Args:
        input (Tensor): Input Tensor.
        dims (tuple(int)): The order of the dimensions. Permute rearranges the `input` according
            to the order of the `dims`.

    Returns:
        Tensor, has the same dimension as input tensor, with `axis` suitably permuted.

    Raises:
        ValueError: If `dims` is None.
        ValueError: If the number of elements of `dims` is not equal to `input` ndim.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> input_x = Tensor(np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]]), mindspore.float32)
        >>> input_perm = (0, 2, 1)
        >>> print(mint.permute(input_x, input_perm))
        [[[ 1.  4.]
          [ 2.  5.]
          [ 3.  6.]]
         [[ 7. 10.]
          [ 8. 11.]
          [ 9. 12.]]]
    """
    return ops.functional.permute(input, dims)


def split(tensor, split_size_or_sections, dim=0):
    """
    Splits the Tensor into chunks along the given dim.

    Args:
        tensor (Tensor): A Tensor to be divided.
        split_size_or_sections (Union[int, tuple(int), list(int)]):
            If `split_size_or_sections` is an int type, `tensor` will be split into equally sized chunks,
            each chunk with size `split_size_or_sections`. Last chunk will be smaller than `split_size_or_sections`
            if `tensor.shape[dim]` is not divisible by `split_size_or_sections`.
            If `split_size_or_sections` is a list type, then `tensor` will be split into len(split_size_or_sections)
            chunks with sizes `split_size_or_sections` along the given `dim`.
        dim (int, optional): The dim along which to split. Default: ``0`` .

    Returns:
        A tuple of sub-tensors.

    Raises:
        TypeError: If argument `tensor` is not Tensor.
        TypeError: If argument `dim` is not int.
        ValueError: If argument `dim` is out of range of [-tensor.ndim, tensor.ndim).
        TypeError: If each element in `split_size_or_sections` is not integer.
        TypeError: If argument `split_size_or_sections` is not int, tuple(int) or list(int).
        ValueError: The sum of `split_size_or_sections` is not equal to tensor.shape[dim].

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import numpy as np
        >>> from mindspore import ops, Tensor
        >>> input_x = np.arange(9).astype("float32")
        >>> output = ops.split(Tensor(input_x), 3)
        >>> print(output)
        (Tensor(shape=[3], dtype=Float32, value= [ 0.00000000e+00,  1.00000000e+00,  2.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 3.00000000e+00,  4.00000000e+00,  5.00000000e+00]),
         Tensor(shape=[3], dtype=Float32, value= [ 6.00000000e+00,  7.00000000e+00,  8.00000000e+00]))
    """
    return ops.function.array_func.split_ext(tensor, split_size_or_sections, dim)


def sqrt(input):
    r"""
    Returns sqrt of a tensor element-wise.

    .. math::

        out_{i} = \sqrt{input_{i}}

    Args:
        input (Tensor): The input tensor with a dtype of number.Number.

    Returns:
        Tensor, has the same shape as the `input`.

    Raises:
        TypeError: If `input` is not a Tensor.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> input = Tensor(np.array([1.0, 4.0, 9.0]), mindspore.float32)
        >>> output = mint.sqrt(input)
        >>> print(output)
        [1. 2. 3.]
    """
    return ops.auto_generate.sqrt(input)


def squeeze(input, dim):
    r"""
    Return the Tensor after deleting the dimension of size 1 in the specified `dim`.

    If :math:`dim=()`, it will remove all the dimensions of size 1.
    If `dim` is specified, it will remove the dimensions of size 1 in the given `dim`.
    For example, if the dimension is not specified :math:`dim=()`, input shape is (A, 1, B, C, 1, D),
    then the shape of the output Tensor is (A, B, C, D). If the dimension is specified, the squeeze operation
    is only performed in the specified dimension. If input shape is (A, 1, B), when :math:`dim=0` or :math:`dim=2`,
    the input tensor is not changed, while when :math:`dim=1`, the input tensor shape is changed to (A, B).

    Note:
        - Please note that in dynamic graph mode, the output Tensor will share data with the input Tensor,
          and there is no Tensor data copy process.
        - The dimension index starts at 0 and must be in the range `[-input.ndim, input.ndim]`.
        - In GE mode, only support remove dimensions of size 1 from the shape of input tensor.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): Used to calculate Squeeze. The shape of tensor is :math:`(x_1, x_2, ..., x_R)`.
        dim (Union[int, tuple(int)]): Specifies the dimension indexes of shape to be removed, which will
            remove all the dimensions of size 1 in the given dim parameter. If specified, it must be int32 or int64.

    Returns:
        Tensor, the shape of tensor is :math:`(x_1, x_2, ..., x_S)`.

    Raises:
        TypeError: If `input` is not a tensor.
        TypeError: If `dim` is not an int, tuple.
        TypeError: If `dim` is a tuple whose elements are not all int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> input = Tensor(np.ones(shape=[3, 2, 1]), mindspore.float32)
        >>> output = mint.squeeze(input, 2)
        >>> print(output)
        [[1. 1.]
         [1. 1.]
         [1. 1.]]
    """
    return squeeze_impl(input, dim)


def sub(input, other, *, alpha=1):
    r"""
    Subtracts scaled other value from input Tensor.

    .. math::

        out_{i} = input_{i} - alpha \times other_{i}

    Note:
        - When the two inputs have different shapes,
          they must be able to broadcast to a common shape.
        - The two inputs and alpha comply with the implicit type conversion rules to make the data types
          consistent.

    Args:
        input (Union[Tensor, number.Number, bool]): The first input is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_.
        other (Union[Tensor, number.Number, bool]): The second input, is a number.Number or
            a bool or a tensor whose data type is
            `number <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_ or
            `bool_ <https://www.mindspore.cn/docs/en/master/api_python/mindspore/mindspore.dtype.html>`_.

    Keyword Args:
        alpha (number.Number, optional): A scaling factor applied to `other`, default ``1``.

    Returns:
        Tensor with a shape that is the same as the broadcasted shape of the input `input` and `other`,
        and the data type is the one with higher precision or higher digits among the two inputs and alpha.

    Raises:
        TypeError: If the type of `input`, `other`, or `alpha` is not one of the following: Tensor, number.Number, bool.
        TypeError: If `alpha` is of type float but `input` and `other` are not of type float.
        TypeError: If `alpha` is of type bool but `input` and `other` are not of type bool.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor
        >>> from mindspore import mint
        >>> x = Tensor(np.array([4, 5, 6]).astype(np.float32))
        >>> y = Tensor(1, mindspore.int32)
        >>> alpha = 0.5
        >>> output = mint.sub(x, y, alpha=alpha)
        >>> print(output)
        [3.5 4.5 5.5]
        >>> # the data type of x is float32, the data type of y is int32,
        >>> # alpha is a float, and the output is the data format of higher precision float32.
        >>> print(output.dtype)
        Float32
    """
    return ops.auto_generate.sub_ext(input, other, alpha)


def swapaxes(input, axis0, axis1):
    '''
    Alias for :func:`mindspore.mint.transpose` . The `input` corresponds to the `input` in the reference interface,
    and the parameters `axis0` and `axis1` correspond to `dim0` and `dim1` in the reference interface respectively.

    For more details, see :func:`mindspore.mint.transpose` .

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Examples:
        >>> import numpy as np
        >>> from mindspore import mint
        >>> from mindspore import Tensor
        >>> input = Tensor(np.ones((2,3,4), dtype=np.float32))
        >>> output = mint.swapaxes(input, 0, 2)
        >>> print(output.shape)
        (4, 3, 2)
    '''
    return transpose(input, axis0, axis1)


def unique_consecutive(input, return_inverse=False, return_counts=False, dim=None):
    r"""
    Returns the elements that are unique in each consecutive group of equivalent elements in the input tensor.

    When `return_inverse=True` , it returns a tensor containing the indices of the elements in the input tensor
    within the output tensor.

    When `return_counts=True` , it returns a tensor representing the number of occurrences of each output element
    in the input.

    .. warning::
        This is an experimental API that is subject to change or deletion.

    Args:
        input (Tensor): The input tensor.
        return_inverse (bool, optional): Whether to return the index of where the element in the original input
            maps to the position in the output. Default: ``False`` .
        return_counts (bool, optional): Whether to return the counts of each unique element. Default: ``False`` .
        dim (int, optional): The dimension to apply unique. If ``None`` , the unique of the flattened input is
            returned. If specified, it must be int32 or int64. Default: ``None`` .

    Returns:
        A tensor or a tuple of tensors containing tensor objects (`output`, `inverse_indices`, `counts`).

        - **output** (Tensor): the output tensor has the same type as `input` and represents the output list of
          unique scalar elements.
        - **inverse_indices** (Tensor, optional): if `return_inverse` is True, there will be an additional returned
          tensor `inverse_indices`. `inverse_indices` has the same shape as `input` and represents the index of where
          the element in the original input maps to the position in the output.
        - **counts** (Tensor, optional):  if `return_counts` is True, there will be an additional returned tensor
          `counts`. `counts` has the same shape as `output` or `output.shape[dim]` if dim was specified and represents
          the number of occurrences for each unique value or tensor.

    Raises:
        TypeError: If `input` is not a Tensor.
        TypeError: If dtype of `input` is not supported.
        TypeError: If `return_inverse` is not a bool.
        TypeError: If `return_counts` is not a bool.
        TypeError: If `dim` is not an int.
        ValueError: If `dim` is not in the range of :math:`[-ndim, ndim-1]`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, mint
        >>> from mindspore import dtype as mstype
        >>> x = Tensor(np.array([1, 1, 2, 2, 3, 1, 1, 2]), mstype.int64)
        >>> output, inverse_indices, counts = mint.unique_consecutive(x, True, True, None)
        >>> print(output)
        [1 2 3 1 2]
        >>> print(inverse_indices)
        [0 0 1 1 2 3 3 4]
        >>> print(counts)
        [2 2 1 2 1]
    """

    return ops.function.array_func.unique_consecutive(input, return_inverse, return_counts, dim)


def zeros(size, *, dtype=None):
    """
    Creates a tensor filled with 0 with shape described by `size` and fills it with value 0 in type of `dtype`.

    Args:
        size (Union[tuple[int], list[int], int, Tensor]): The specified shape of output tensor. Only positive integer or
            tuple or Tensor containing positive integers are allowed. If it is a Tensor,
            it must be a 0-D or 1-D Tensor with int32 or int64 dtypes.

    Keyword Args:
        dtype (:class:`mindspore.dtype`, optional): The specified type of output tensor. If `dtype` is ``None`` ,
            mindspore.float32 will be used. Default: ``None`` .

    Returns:
        Tensor, whose dtype and size are defined by input.

    Raises:
        TypeError: If `size` is neither an int nor a tuple/list/Tensor of int.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore
        >>> from mindspore import mint
        >>> output = mint.zeros((2, 2), dtype=mindspore.float32)
        >>> print(output)
        [[0. 0.]
         [0. 0.]]
    """
    return ops.auto_generate.zeros(size, dtype)


def fix(input):
    """
    Alias for :func:`mindspore.mint.trunc` .

    For more details, see :func:`mindspore.mint.trunc` .

    Supported Platforms:
        ``Ascend``
    """
    return trunc(input)


def scatter(input, dim, index, src):
    """
    Update the value in `src` to `input` according to the specified index.
    For a 3-D tensor, the output will be:

    .. code-block::

        output[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0

        output[i][index[i][j][k]][k] = src[i][j][k]  # if dim == 1

        output[i][j][index[i][j][k]] = src[i][j][k]  # if dim == 2

    .. note::
        The backward is supported only for the case `src.shape == index.shape` when `src` is a tensor.

    Args:
        input (Tensor): The target tensor. The rank of `input` must be at least 1.
        dim (int): Which axis to scatter. Accepted range is [-r, r) where r = rank(input).
        index (Tensor): The index to do update operation whose data must be positive number with type of mindspore.int32
            or mindspore.int64. Same rank as `input` . And accepted range is [-s, s) where s is the size along axis.
        src (Tensor, float): The data doing the update operation with `input`. Can be a tensor with the same data type
            as `input` or a float number to scatter.

    Returns:
        Tensor, has the same shape and type as `input` .

    Raises:
        TypeError: If `index` is neither int32 nor int64.
        ValueError: If rank of any of `input` , `index` and `src` is less than 1.
        ValueError: If the rank of `src` is not equal to the rank of `input` .
        TypeError: If the data types of `input` and `src` have different dtypes.
        RuntimeError: If `index` has negative elements.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore as ms
        >>> from mindspore import Tensor, mint
        >>> input = Tensor(np.array([[1, 2, 3, 4, 5]]), dtype=ms.float32)
        >>> src = Tensor(np.array([[8, 8]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[2, 4]]), dtype=ms.int64)
        >>> out = mint.scatter(input=input, dim=1, index=index, src=src)
        >>> print(out)
        [[1. 2. 8. 4. 8.]]
        >>> input = Tensor(np.zeros((5, 5)), dtype=ms.float32)
        >>> src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[0, 0, 0], [2, 2, 2], [4, 4, 4]]), dtype=ms.int64)
        >>> out = mint.scatter(input=input, dim=0, index=index, src=src)
        >>> print(out)
        [[1. 2. 3. 0. 0.]
        [0. 0. 0. 0. 0.]
        [4. 5. 6. 0. 0.]
        [0. 0. 0. 0. 0.]
        [7. 8. 9. 0. 0.]]
        >>> input = Tensor(np.zeros((5, 5)), dtype=ms.float32)
        >>> src = Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), dtype=ms.float32)
        >>> index = Tensor(np.array([[0, 2, 4], [0, 2, 4], [0, 2, 4]]), dtype=ms.int64)
        >>> out = mint.scatter(input=input, dim=1, index=index, src=src)
        >>> print(out)
        [[1. 0. 2. 0. 3.]
        [4. 0. 5. 0. 6.]
        [7. 0. 8. 0. 9.]
        [0. 0. 0. 0. 0.]
        [0. 0. 0. 0. 0.]]
    """
    return ops.function.array_func.scatter(input, dim, index, src)


def cdist(x1, x2, p=2.0, compute_mode='use_mm_for_euclid_dist_if_necessary'):
    """
    Computes p-norm distance between each pair of row vectors of two input Tensors.

    .. warning::
        This is an experimental optimizer API that is subject to change.

    Note:
        On Ascend, the supported dtypes are float16 and float32.
        On CPU, the supported dtypes are float16 and float32.
        On GPU, the supported dtypes are float32 and float64.

    Args:
        x1 (Tensor): Input tensor of shape :math:`(B, P, M)`.
            Letter :math:`B` represents 0 or positive int number.
            When :math:`B` is equal to 0, it means this dimension can be ignored,
            i.e. shape of the tensor is :math:`(P, M)`.
        x2 (Tensor): Input tensor of shape :math:`(B, R, M)`, has the same dtype as `x1`.
        p (float, optional): P value for the p-norm distance to calculate between each
            vector pair, P >= 0. Default: ``2.0`` .
        compute_mode (string, optional): Specify the cumpute mode. Setting this parameter currently has no effect.
            Default: ``'use_mm_for_euclid_dist_if_necessary'`` .

    Returns:
        Tensor, p-norm distance, has the same dtype as `x1`, its shape is :math:`(B, P, R)`.

    Raises:
        TypeError: If `x1` or `x2` is not Tensor.
        TypeError: If dtype of `x1` or `x2` is not listed in the "Note" above.
        TypeError: If `p` is not float32.
        ValueError: If `p` is negative.
        ValueError: If dimension of `x1` is not the same as `x2`.
        ValueError: If dimension of `x1` or `x2` is neither 2 nor 3.
        ValueError: If the batch dim of `x1` and `x2` can not broadcast.
        ValueError: If the number of columns of `x1` is not the same as that of `x2`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> x = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(np.float32))
        >>> y = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(np.float32))
        >>> output = ops.cdist(x, y, 2.0)
        >>> print(output)
        [[[2.8284273 2.8284273]
          [1.4142137 1.4142137]]]
    """
    return cdist_(x1, x2, p)


__all__ = [
    'conv2d',
    'full',
    'ones_like',
    'zeros_like',
    'abs',
    'clone',
    'erf',
    'where',
    'isclose',
    'empty',
    'empty_like',
    'full_like',
    # 1
    'div',
    'divide',
    'topk',
    'roll',
    # 2
    'sin',
    # 3
    'clamp',
    'xlogy',
    'fmod',
    # 4
    'sinc',
    'sinh',
    'cosh',
    # 5
    'cumsum',
    # 6
    'stack',
    # 7
    'zeros',
    # 8
    'transpose',
    'swapaxes',
    "batch_norm_elemt",
    "batch_norm_gather_stats_with_counts",
    "batch_norm_stats",
    # 9
    'squeeze',
    # 10
    'ne',
    'not_equal',
    # 11
    'unsqueeze',
    # 12
    "repeat_interleave",
    # 13
    "flip",
    # 14
    'mv',
    # 15
    'flatten',
    # 16
    'matmul',
    'bmm',
    # 17
    'mean',
    # 18
    'sum',
    # 19
    'log',
    # 20
    'prod',
    # 21
    'mul',
    # 22
    'cumprod',
    # 23
    'exp2',
    # 24
    'cdist',
    # 25
    'greater',
    'gt',
    # 26
    'eq',
    # 27
    'reciprocal',
    # 28
    'exp',
    # 29
    'sqrt',
    # 30
    'searchsorted',
    # 31
    'cummax',
    'cummin',
    'einsum',
    'sub',
    # 33
    'split',
    # 34

    # 35
    'erfinv',
    # 36

    # 37
    'nonzero',
    # 38

    # 39

    # 40
    'any',
    # 41
    'add',
    # 42
    'argmax',
    # 43
    'cat',
    # 44
    'cos',
    # 45
    'concat',
    # 46
    'bitwise_and',
    'bitwise_or',
    'bitwise_xor',
    # 47
    'max',
    # 48
    'min',
    # 49
    'baddbmm',
    # 50
    'tile',
    # 51
    'permute',
    # 52

    # 53

    # 54
    'normal',
    # 55
    'cross',
    # 56
    'norm',
    # 57
    'broadcast_to',
    # 58
    'greater_equal',
    # 59
    'square',
    # 60
    'all',
    # 61
    'rsqrt',
    # 62
    'maximum',
    # 63
    'minimum',
    # 64
    'ravel',
    # 65
    'logical_and',
    # 66
    'logical_not',
    # 67
    'logical_or',
    # 68
    'logical_xor',
    # 69
    'less_equal',
    'le',
    # 70
    'negative',
    'neg',
    # 71
    'isfinite',
    # 72

    # 73
    'ceil',
    # 74
    'sort',
    # 75
    'less',
    'lt',
    # 76
    'pow',
    # 77

    # 78
    'arange',

    # 79

    # 80

    # 81
    'index_select',
    # 82

    # 83
    'narrow',
    # 84
    'masked_select',

    # 86
    'select',

    # 87

    # 88
    'chunk',
    # 89
    'argsort',
    # 90
    'isinf',
    # 91

    # 92
    'polar',
    # 93

    # 94
    'tanh',
    # 95

    # 96

    # 97

    # 98

    # 99

    # 100

    # 101

    # 102

    # 103

    # 104

    # 105

    # 106

    # 107

    # 108

    # 109
    'argmin',
    # 110
    'softmax',
    # 111

    # 112

    # 113

    # 114

    # 115

    # 116

    # 117

    # 118

    # 119

    # 120
    'isneginf',
    # 121

    # 122

    # 123
    'var',

    # 151
    'acos',
    'arccos',
    # 152
    'acosh',
    'arccosh',
    # 153

    # 154

    # 155

    # 156

    # 157
    'scatter',
    # 172
    'asin',
    'arcsin',
    # 173
    'asinh',
    'arcsinh',
    # 174
    'atan',
    'arctan',
    # 175
    'atanh',
    'arctanh',
    # 176
    'atan2',
    'arctan2',

    # 177
    'round',

    # 182
    'bernoulli',

    # 207
    'expm1',
    # 204
    'erfc',
    # 208
    'eye',
    # 239
    'lerp',
    # 256
    'median',
    'randperm',
    'rand',
    'rand_like',
    'randn',
    'randn_like',
    'randint',
    'randint_like',
    # 210
    'floor',
    # 231
    'inverse',
    # 244
    'log1p',
    # 261
    'multinomial',
    # 275
    'remainder',
    # 285
    'scatter_add',
    # 289
    'sign',
    # 301
    'tan',
    # 303
    'trace',
    'gcd',
    'reshape',
    'outer',
    # 304
    'tril',

    # 305
    'triu',

    # 308
    'mm',

    # 382
    'dstack',

    # 406
    'allclose',

    # 501
    'addbmm',

    # 502
    'addmm',

    # 505
    'addmv',

    # 510
    'amax',

    # 511
    'amin',

    # 520
    'bincount',

    # 521
    'bitwise_not',

    # 526
    'dot',

    # 533
    'frac',

    # 538
    'histc',

    # 552
    'log10',

    # 553
    'logaddexp',

    # 557
    'logsumexp',

    # 582
    'std_mean',

    # 588
    'var_mean',

    # 586
    'unique_consecutive',

    # 610
    'nan_to_num',

    # 613
    'nansum',

    # 664
    'meshgrid',

    # 695
    'count_nonzero',

    # 697
    'float_power',

    # 708
    'std',

    # 887
    'log2',

    # 889
    'isnan',

    # 1007
    't',

    # 1023
    'unbind',

    # 1100
    'diff',
]

__all__.extend(functional.__all__)
__all__.extend(nn.__all__)
__all__.extend(optim.__all__)
__all__.extend(linalg.__all__)
__all__.extend(special.__all__)
__all__.extend(distributed.__all__)
