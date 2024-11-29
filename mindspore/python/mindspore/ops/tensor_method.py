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
"""Tensor method for overload."""

from mindspore import _checkparam as validator
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.composite.multitype_ops import _compile_utils as utils
from mindspore.ops.composite.multitype_ops._compile_utils import sequence_to_tensor
from mindspore.ops.auto_generate.gen_ops_prim import (
    inplace_scatter_src_op, inplace_scatter_src_reduce_op, inplace_scatter_value_op, inplace_scatter_value_reduce_op
)
# 1 common import

# 2 common import
from mindspore import Tensor
# 3 common import
from mindspore.common import dtype as mstype
# 4 common import
from mindspore.common import COOTensor
# 5 common import

# 6 common import

# 7 common import


# 1 to
from mindspore.ops.auto_generate import cast
# 2 masked_fill

# 3 abs
from mindspore.ops.auto_generate import abs
# 4 __abs__

# 5 add
from mindspore.ops.auto_generate import add_ext, add
# 6 all
from mindspore.ops.auto_generate import all
# 7 allclose

# 8 any
from mindspore.ops.function.math_func import any
# 9 arctan2
from mindspore.ops.function.math_func import arctan2
# 10 argmax
from mindspore.ops.function.array_func import argmax
# 11 argmin
from mindspore.ops.function.math_func import argmin
# 12 argsort

# 13 atan2
from mindspore.ops.function.math_func import atan2
# 14 bfloat16

# 15 bmm

# 16 bool

# 17 broadcast_to

# 18 byte

# 19 ceil
from mindspore.ops.function.math_func import ceil
# 20 chunk
from mindspore.ops.function.array_func import chunk
# 21 clamp
from mindspore.ops.auto_generate import clamp_tensor, clamp_scalar
# 22 clip

# 23 cos
from mindspore.ops.function.math_func import cos
# 24 cumprod

# 25 cumsum
from mindspore.ops.function.math_func import cumsum
# 26 dim

# 27 div
from mindspore.ops.function.math_func import div
# 28 divide

# 29 eq
from mindspore.ops.function.math_func import eq
# 30 erf
from mindspore.ops.auto_generate import erf
# 31 exp
from mindspore.ops.auto_generate import exp
# 32 expand

# 33 expand_as

# 34 flatten
from mindspore.ops.function.array_func import flatten

# 35 flip

# 36 float

# 37 floor
from mindspore.ops.function.math_func import floor
# 38 gather
from mindspore.ops.auto_generate import gather
from mindspore.ops.function.array_func import gather_ext
# 39 greater
from mindspore.ops.function.math_func import greater
# 40 greater_equal

# 41 gt

# 42 half

# 43 index_put

# 44 index_select
from mindspore.ops.function.array_func import index_select
# 45 int

# 46 inverse

# 47 is_contiguous

# 48 isclose

# 49 isfinite
from mindspore.ops.auto_generate import isfinite
# 50 isnan

# 51 item

# 52 le
from mindspore.ops.function.math_func import le
# 53 less

# 54 less_equal

# 55 log

# 56 log2

# 57 logical_and
from mindspore.ops.function.math_func import logical_and
# 58 logical_not
from mindspore.ops.function.math_func import logical_not
# 59 logical_or
from mindspore.ops.function.math_func import logical_or
# 60 long

# 61 lt

# 62 masked_fill
from mindspore.ops.auto_generate import masked_fill
# 63 masked_select

# 64 matmul
from mindspore.ops.auto_generate import matmul_ext
# 65 max
from mindspore.ops.auto_generate import max_
from mindspore.ops.function.array_func import max as max_func
# 66 maximum

# 67 mean
from mindspore.ops.auto_generate import mean_ext
from mindspore.ops.function.math_func import mean
# 68 min
from mindspore.ops.auto_generate import min_
from mindspore.ops.function.array_func import min as min_func
# 69 minimum

# 70 mul

# 71 nan_to_num

# 72 narrow

# 73 ne

# 74 neg

# 75 negative

# 76 nonzero

# 77 norm

# 78 numel

# 79 numpy

# 80 outer

# 81 permute

# 82 pow
from mindspore.ops.auto_generate import pow
# 83 prod
from mindspore.ops.auto_generate import prod_ext
# 84 reciprocal
from mindspore.ops.function.math_func import reciprocal
# 85 remainder
from mindspore.ops.function.math_func import remainder
# 86 repeat

# 87 repeat_interleave
from mindspore.ops.function.array_func import repeat_interleave, repeat_interleave_ext
# 88 reshape
from mindspore.ops.auto_generate import reshape
# 89 round
from mindspore.ops.function.math_func import round
# 90 rsqrt
from mindspore.ops.auto_generate import rsqrt
# 91 scatter

# 92 scatter_add

# 93 select
from mindspore.ops.auto_generate import select, select_ext
# 94 sigmoid
from mindspore.ops.auto_generate import sigmoid
# 95 sin
from mindspore.ops.auto_generate import sin
# 96 size

# 97 sort
from mindspore.ops.function.array_func import sort
# 98 split
from mindspore.ops.function.array_func import split
# 99 sqrt
from mindspore.ops.auto_generate import sqrt
# 100 square
from mindspore.ops.auto_generate import square
# 101 squeeze

# 102 std

# 103 sub
from mindspore.ops.auto_generate import sub
# 104 sum
from mindspore.ops.function.math_func import sum
# 105 swapaxes

# 106 t

# 107 tanh
from mindspore.ops.auto_generate import tanh
# 108 tile
from mindspore.ops.operations.manually_defined import tile
# 109 tolist

# 110 topk
from mindspore.ops.function.array_func import topk
# 111 transpose

# 112 tril
from mindspore.ops.function.array_func import tril
# 113 trunc

# 114 type

# 115 type_as

# 116 unbind

# 117 unfold

# 118 unique

# 119 unsqeeze

# 120 view

# 121 contiguous

# 122 where
from mindspore.ops.function.array_func import where as where_func


# 123 div_

# 124 fill_

# 125 floor_

# 126 masked_fill_

# 127 mul_

# 128 normal_

# 129 requires_grad_

# 130 sub_

# 131 uniform_

# 132 absolute

# 133 bincount

# 134 diff

# 135 double

# 136 lcm

# 137 mm

# 138 ravel

# 139 nelement

# 140 stride

# 141 indices

# 142 view_as

# 143 values

# 144 index_copy

# 145 element_size

# 146 gcd

# 147 isinf

# 148 not_equal

# 149 triu

# 150 __eq__

# 151 fmod

# 152

# 153

# 154 isneginf
from mindspore.ops.auto_generate import isneginf_ext

# 155

# 156

# 157

# 158

# 159

# 160

# 161

# 162

from mindspore.ops.auto_generate import clone
from mindspore.ops.function.array_func import new_ones
from mindspore.ops.function.array_func import new_zeros

########################################functions########################################


# 1 to
def tensor_to(input, dtype):
    return cast(input, dtype)


# 2 masked_fill
def tensor_masked_fill(input_x, mask, value):
    return masked_fill(input_x, mask, value)


# 3 abs
def tensor_abs(input):
    return abs(input)


# 4 __abs__

# 5 add
def tensor_add_ext(input, other, *, alpha=1):
    return add_ext(input, other, alpha=alpha)


def deprecated_tensor_add(input, other):
    if isinstance(other, COOTensor):
        return other + input
    if isinstance(other, (tuple, list)):
        other = sequence_to_tensor(other, F.dtype(input))
    return add(input, other)


# 6 all
def tensor_all(x, axis=None, keep_dims=False):
    return all(x, axis, keep_dims)


def deprecated_tensor_all(x, dim=None, keepdim=False):
    return all(x, dim, keepdim)


# 7 allclose

# 8 any
def tensor_any(x, axis=None, keep_dims=False):
    if axis is None:
        axis = ()
    return any(x, axis, keep_dims)


# 9 arctan2
def tensor_arctan2(input, other):
    return arctan2(input, other)


# 10 argmax
def tensor_argmax(input, dim=None, keepdim=False):
    return argmax(input, dim, keepdim)


def deprecated_tensor_argmax(input, axis=None, keepdims=False):
    return argmax(input, axis, keepdims)


# 11 argmin
def tensor_argmin(input, dim=None, keepdim=False):
    return argmin(input, dim, keepdim)


def deprecated_tensor_argmin(input, axis=None, keepdims=False):
    return argmin(input, axis, keepdims)


# 12 argsort

# 13 atan2
def tensor_atan2(input, other):
    return atan2(input, other)


# 14 bfloat16

# 15 bmm

# 16 bool

# 17 broadcast_to

# 18 byte

# 19 ceil
def tensor_ceil(input):
    return ceil(input)


# 20 chunk
def deprecated_tensor_chunk(input, chunks, axis=0):
    return chunk(input, chunks, axis)


def tensor_chunk(input, chunks, dim=0):
    return chunk(input, chunks, dim)


# 21 clamp
def tensor_clamp_tensor(input, min=None, max=None):
    return clamp_tensor(input, min, max)


def tensor_clamp_scalar(input, min=None, max=None):
    return clamp_scalar(input, min, max)


# 22 clip

# 23 cos
def tensor_cos(input):
    return cos(input)


# 24 cumprod

# 25 cumsum
def deprecated_tensor_cumsum(x, axis=None, dtype=None):
    r"""
    For details, please refer to :func:`mindspore.ops.cumsum`.
    """
    original_dtype = x.dtype
    # If original tensor is int, and has precision less then int32, convert to int32
    if x.dtype in (mstype.bool_, mstype.int8, mstype.int16, mstype.uint8, mstype.int16):
        x = x.astype(mstype.int32)
    if axis is None:
        x = x.ravel()
        axis = 0
    validator.check_axis_in_range(axis, x.ndim)
    if dtype is not None and original_dtype != dtype:
        return cumsum(x, axis).astype(dtype, copy=False)
    return cumsum(x, axis)


def tensor_cumsum(input, dim, *, dtype=None):
    return deprecated_tensor_cumsum(input, dim, dtype)


# 26 dim

# 27 div
def tensor_div(input, value, *, rounding_mode=None):
    return div(input, value, rounding_mode=rounding_mode)


# 28 divide

# 29 eq
def tensor_eq(input, other):
    return eq(input, other)


# 30 erf
def tensor_erf(input):
    return erf(input)


# 31 exp
def tensor_exp(input):
    return exp(input)


# 32 expand

# 33 expand_as

# 34 flatten
def deprecated_tensor_flatten(input, order='C', *, start_dim=0, end_dim=-1):
    return flatten(input, order, start_dim=start_dim, end_dim=end_dim)


def tensor_flatten(input, start_dim=0, end_dim=-1):
    return flatten(input, start_dim=start_dim, end_dim=end_dim)


# 35 flip

# 36 float

# 37 floor
def tensor_floor(input):
    return floor(input)


# 38 gather
def tensor_gather_ext(input, dim, index):
    return gather_ext(input, dim, index)


def deprecated_tensor_gather(input, input_indices, axis, batch_dims=0):
    r"""
    For details, please refer to :func:`mindspore.ops.gather`.
    """
    validator.check_is_int(axis, 'axis')
    validator.check_is_int(batch_dims, "batch_dims")
    return gather(input, input_indices, axis, batch_dims)


# 39 greater
def tensor_greater(input, other):
    return greater(input, other)


# 40 greater_equal

# 41 gt

# 42 half

# 43 index_put

# 44 index_select
def tensor_index_select(input, dim, index):
    return index_select(input, dim, index)


def deprecated_tensor_index_select(input, axis, index):
    return index_select(input, axis, index)


# 45 int

# 46 inverse

# 47 is_contiguous

# 48 isclose

# 49 isfinite
def tensor_isfinite(input):
    return isfinite(input)


# 50 isnan

# 51 item

# 52 le
def tensor_le(input, other):
    return le(input, other)


# 53 less
def tensor_less(input, other):
    return F.less(input, other)


# 54 less_equal

# 55 log
def tensor_log(input):
    return F.log(input)


# 56 log2

# 57 logical_and
def tensor_logical_and(input, other):
    return logical_and(input, other)


# 58 logical_not
def tensor_logical_not(input):
    return logical_not(input)


# 59 logical_or
def tensor_logical_or(input, other):
    return logical_or(input, other)


# 60 long

# 61 lt

# 62 masked_fill

# 63 masked_select
def tensor_masked_select(tensor, mask):
    return F.masked_select(tensor, mask)


# 64 matmul
def tensor_matmul(input, mat2):
    return matmul_ext(input, mat2)


def deprecated_tensor_matmul(input, tensor2):
    return F.matmul(input, tensor2)


# 65 max
def tensor_max(input):
    return max_(input)


def deprecated_tensor_max(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False):
    if isinstance(axis, (list, tuple)):
        reduce_max = P.ReduceMax
        maximum = F.maximum
        return utils.reduce_(input, reduce_max(keepdims), cmp_fn=maximum, axis=axis, keepdims=keepdims,
                             initial=initial, where=where)
    values, indices = max_func(input, axis, keepdims, initial=initial, where=where)
    if not return_indices:
        return values
    return values, indices


# 66 maximum
def tensor_maximum(input, other):
    return F.maximum(input, other)


# 67 mean
def tensor_mean_ext(input, axis=None, keep_dims=False, dtype=None):
    return mean_ext(input, axis, keep_dims, dtype)


def deprecated_tensor_mean(input, axis=None, keep_dims=False):
    return mean(input, axis, keep_dims)


# 68 min
def tensor_min(input):
    return min_(input)


def deprecated_tensor_min(input, axis=None, keepdims=False, *, initial=None, where=True, return_indices=False):
    if isinstance(axis, (list, tuple)):
        reduce_min = P.ReduceMin
        minimum = F.minimum
        return utils.reduce_(input, reduce_min(keepdims), cmp_fn=minimum, axis=axis, keepdims=keepdims,
                             initial=initial, where=where)
    values, indices = min_func(input, axis, keepdims, initial=initial, where=where)
    if not return_indices:
        return values
    return values, indices


# 69 minimum
def tensor_minimum(input, other):
    return F.minimum(input, other)


# 70 mul
def tensor_mul(input, other):
    return F.mul(input, other)


# 71 nan_to_num
def tensor_nan_to_num(input, nan=0.0, posinf=None, neginf=None):
    return F.nan_to_num(input, nan, posinf, neginf)


# 72 narrow

# 73 ne
def tensor_ne(input, other):
    return F.ne(input, other)


# 74 neg
def tensor_neg(input):
    return F.neg(input)


# 75 negative

# 76 nonzero

# 77 norm

# 78 numel

# 79 numpy

# 80 outer

# 81 permute

# 82 pow
def tensor_pow(input, exponent):
    return pow(input, exponent)


# 83 prod
def tensor_prod(input, axis=None, keep_dims=False, dtype=None):
    return prod_ext(input, axis, keep_dims, dtype)


def deprecated_tensor_prod(input, dim=None, keepdim=False, dtype=None):
    return prod_ext(input, dim, keepdim, dtype)


# 84 reciprocal
def tensor_reciprocal(input):
    return reciprocal(input)


# 85 remainder
def tensor_remainder(input, other):
    return remainder(input, other)


def deprecated_tensor_remainder(input, divisor):
    return remainder(input, divisor)


# 86 repeat

# 87 repeat_interleave
def deprecated_tensor_repeat_interleave(input, repeats, dim=None):
    return repeat_interleave(input, repeats, dim)


def tensor_repeat_interleave_ext(input, repeats, dim=None, *, output_size=None):
    return repeat_interleave_ext(input, repeats, dim, output_size)


# 88 reshape
def tensor_reshape(input, *shape):
    new_shape = validator.check_reshape_shp(shape)
    return reshape(input, new_shape)


# 89 round
def tensor_round(input, decimals=0):
    return round(input, decimals=decimals)


# 90 rsqrt
def tensor_rsqrt(input):
    return rsqrt(input)


# 91 scatter

# 92 scatter_add

# 93 select
def tensor_select_ext(input, dim, index):
    return select_ext(input, dim, index)


def deprecated_tensor_select(input, condition, y):
    r"""
    For details, please refer to :func:`mindspore.ops.select`.
    """
    if not isinstance(condition, Tensor):
        raise TypeError(f"For 'Tensor.select', the argument 'condition' should be Tensor,"
                        f" but got {type(condition)}.")
    if not isinstance(y, (Tensor, int, float)):
        raise TypeError(f"For 'Tensor.select', the argument 'y' should be Tensor, int or float,"
                        f" but got {type(y)}.")
    if isinstance(y, int) and input.dtype != mstype.int32:
        raise TypeError(f"For 'Tensor.select', if the argument 'y' is int,"
                        f" then the tensor type should be int32 but got {input.dtype}")
    if isinstance(y, float) and input.dtype != mstype.float32:
        raise TypeError(f"For 'Tensor.select', if the argument 'y' is float,"
                        f" then the tensor type should be float32 but got {input.dtype}")
    input_y = y
    if isinstance(y, (int, float)):
        zeros_like = F.zeros_like
        cast_f = F.cast
        input_y = zeros_like(input) + y
        if isinstance(y, int):
            input_y = cast_f(input_y, mstype.int32)
        else:
            input_y = cast_f(input_y, mstype.float32)
    return select(condition, input, input_y)


# 94 sigmoid
def tensor_sigmoid(input):
    return sigmoid(input)


# 95 sin
def tensor_sin(input):
    return sin(input)


# 96 size

# 97 sort
def deprecated_tensor_sort(input_x, axis=-1, descending=False):
    return sort(input_x, axis, descending)


def tensor_sort(input, dim=-1, descending=False, stable=False):
    return sort(input, dim, descending)


# 98 split
def deprecated_tensor_split(input, split_size_or_sections, axis=0):
    return split(input, split_size_or_sections, axis)


# 99 sqrt
def tensor_sqrt(x):
    return sqrt(x)


# 100 square
def tensor_square(input):
    return square(input)


# 101 squeeze

# 102 std

# 103 sub
def deprecated_tensor_sub(input, y):
    if isinstance(y, COOTensor):
        return F.tensor_scatter_sub(input, y.indices, y.values)
    if isinstance(input, (tuple, list)):
        input = sequence_to_tensor(input, F.dtype(y))
    if isinstance(y, (tuple, list)):
        y = sequence_to_tensor(y, F.dtype(input))
    return sub(input, y)


# 104 sum
def deprecated_tensor_sum(input, axis=None, dtype=None, keepdims=False, initial=None):
    if initial is None:
        res = sum(input, axis, keepdims, dtype=dtype)
    else:
        res = sum(input, axis, keepdims, dtype=dtype) + initial
    if dtype is not None and (dtype == mstype.bool_):
        res = res.astype(mstype.bool_)
    return res


# 105 swapaxes

# 106 t

# 107 tanh
def tensor_tanh(input):
    return tanh(input)


# 108 tile
def tensor_tile(input, dims):
    return tile(input, dims)


def deprecated_tensor_tile(input, reps):
    return tile(input, reps)


# 109 tolist

# 110 topk
def tensor_topk(input, k, dim=-1, largest=True, sorted=True):
    return topk(input, k, dim, largest, sorted)


def deprecated_tensor_topk(input, k, dim=None, largest=True, sorted=True):
    return topk(input, k, dim, largest, sorted)


# 111 transpose

# 112 tril
def deprecated_tensor_tril(input, diagonal=0):
    return tril(input, diagonal)


# 113 trunc
def tensor_trunc(input):
    return F.trunc(input)


# 114 type

# 115 type_as

# 116 unbind

# 117 unfold

# 118 unique

# 119 unsqeeze

# 120 view

# 121 contiguous

# 122 where
def tensor_where(condition, input, other):
    return where_func(condition, input, other)


def deprecated_tensor_where(input, condition, y):
    return where_func(condition, input, y)


# 123 div_

# 124 fill_

# 125 floor_

# 126 masked_fill_

# 127 mul_

# 128 normal_

# 129 requires_grad_

# 130 sub_

# 131 uniform_

# 132 absolute

# 133 bincount

# 134 diff

# 135 double

# 136 lcm

# 137 mm

# 138 ravel

# 139 nelement

# 140 stride

# 141 indices

# 142 view_as

# 143 values

# 144 index_copy

# 145 element_size

# 146 gcd

# 147 isinf

# 148 not_equal
def tensor_not_equal(input, other):
    return F.ne(input, other)


# 149 triu
def tensor_triu(input, diagonal=0):
    return F.triu(input, diagonal)

# 150 __eq__

# 151 scatter_

def tensor_inplace_scatter_src(input, dim, index, src):
    return inplace_scatter_src_op(input, dim, index, src)


def tensor_inplace_scatter_src_reduce(input, dim, index, src, *, reduce):
    return inplace_scatter_src_reduce_op(input, dim, index, src, reduce=reduce)


def tensor_inplace_scatter_value(input, dim, index, value):
    return inplace_scatter_value_op(input, dim, index, value)


def tensor_inplace_scatter_value_reduce(input, dim, index, value, *, reduce):
    return inplace_scatter_value_reduce_op(input, dim, index, value, reduce=reduce)


# 152 fmod
def fmod_tensor(input, other):
    return


def fmod_scalar(input, other):
    return

# 153

# 154
def tensor_isneginf(input):
    return isneginf_ext(input)

# 155

# 156

# 157

# 158

# 159

# 160

# 161

# 162

def tensor_clone(input):
    return clone(input)


def tensor_new_ones(input, size, dtype=None):
    return new_ones(input, size, dtype=dtype)


def tensor_new_zeros(input, size, dtype=None):
    return new_zeros(input, size, dtype=dtype)
