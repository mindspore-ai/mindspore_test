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

from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops.composite.multitype_ops import _compile_utils as utils
from mindspore.ops.auto_generate import add, max_
from mindspore.ops.auto_generate import clamp_tensor, clamp_scalar
from mindspore.ops.function.math_func import mean, ceil, cos
from mindspore.ops.function.array_func import argmax
from mindspore.ops.function.array_func import max as max_func


def tensor_clamp_tensor(input, min=None, max=None):
    return clamp_tensor(input, min, max)


def tensor_clamp_scalar(input, min=None, max=None):
    return clamp_scalar(input, min, max)


def tensor_mean(x, axis=None, keep_dims=False, dtype=None):
    return mean(x, axis, keep_dims)


def tensor_argmax(input, dim=None, keepdim=False):
    return argmax(input, dim, keepdim)


def deprecated_tensor_argmax(input, axis=None, keepdims=False):
    return argmax(input, axis, keepdims)


def tensor_add(input, other, alpha=1):
    return add(input, other)


def tensor_max(input):
    return max_(input)


def deprecated_tensor_max(input, axis=None, keepdims=False, *, initial=None, where=None, return_indices=False):
    if isinstance(axis, (list, tuple)):
        reduce_max = P.ReduceMax
        maximum = F.maximum
        return utils.reduce_(input, reduce_max(keepdims), cmp_fn=maximum, axis=axis, keepdims=keepdims,
                             initial=initial, where=where)
    values, indices = max_func(input, axis, keepdims, initial=initial, where=where)
    if not return_indices:
        return values
    return values, indices


def tensor_ceil(input):
    return ceil(input)


def tensor_cos(input):
    return cos(input)
