# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Generate vm_impl function for math ops"""
import copy
import numpy as np

import mindspore.common.dtype as mstype
from mindspore._c_expression import typing
from mindspore.common.tensor import Tensor
from mindspore.ops import operations as P
from mindspore.ops.auto_generate.gen_ops_prim import SubExt
from mindspore.ops.vm_impl_registry import vm_impl_registry as vm_impl_getters
from .vm_interface import vm
from mindspore.ops.auto_generate.gen_ops_prim import AddExt


# pylint: disable=unused-argument

@vm_impl_getters.register(P.FloorMod)
def vm_impl_floor_mod(self):
    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        return Tensor(np.mod(x, y))
    return vm_impl

@vm_impl_getters.register(P.ZerosLike)
def vm_impl_zeroslike(self):
    def vm_impl(x):
        x = x.asnumpy()
        out = np.zeros_like(x)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Zeros)
def vm_impl_zeros(self):
    def vm_impl(x, y):
        out = np.zeros(x, mstype.dtype_to_nptype(typing.type_id_to_type(y)))
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Ones)
def vm_impl_ones(self):
    def vm_impl(x, y):
        out = np.ones(x, mstype.dtype_to_nptype(typing.type_id_to_type(y)))
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Log)
def vm_impl_log(self):
    def vm_impl(x):
        x = x.asnumpy()
        out = np.log(x)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Add)
def vm_impl_tensor_add(self):
    """Generate vm_impl function for TensorAdd."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        return Tensor(np.array(x + y, dtype=x.dtype))

    return vm_impl


@vm_impl_getters.register(AddExt)
def vm_impl_tensor_addext(self):
    """Generate vm_impl function for TensorAddExt."""

    def vm_impl(x, y, alpha=1):
        x = x.asnumpy()
        y = y.asnumpy()
        alpha = np.array(alpha, dtype=x.dtype)
        return Tensor(np.array(x + alpha * y, dtype=x.dtype))

    return vm_impl


# pylint: disable=used-before-assignment
@vm_impl_getters.register(P.LogicalNot)
def vm_impl_logical_not(self):
    def vm_impl(x):
        x = x.asnumpy()
        out = vm.logical_not(x)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.MatMul)
def vm_impl_mat_mul(self):
    """Generate vm_impl function for MatMul."""

    def vm_impl(x, w, transpose_a, transpose_b):
        x = x.asnumpy()
        w = w.asnumpy()
        if transpose_a:
            x = x.transpose()
        if transpose_b:
            w = w.transpose()
        z = x @ w
        return Tensor(z)

    return vm_impl


@vm_impl_getters.register(P.AddN)
def vm_impl_addn(self):
    """Generate vm_impl function for AddN."""

    def vm_impl(inputs):
        added = copy.deepcopy(inputs[0].asnumpy())
        for x in inputs[1:]:
            added += x.asnumpy()
        return Tensor(added)

    return vm_impl


@vm_impl_getters.register(P.Neg)
def vm_impl_neg(self):
    """Generate vm_impl function for Neg."""

    def vm_impl(x):
        x = x.asnumpy()
        return Tensor(-x)

    return vm_impl


@vm_impl_getters.register(P.Sub)
def vm_impl_Sub(self):
    """Generate vm_impl function for Sub."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        return Tensor(x - y)

    return vm_impl


@vm_impl_getters.register(SubExt)
def vm_impl_tensor_SubExt(self):
    """Generate vm_impl function for TensorSubExt."""

    def vm_impl(x, y, alpha=1):
        x = x.asnumpy()
        y = y.asnumpy()
        alpha = np.array(alpha, dtype=x.dtype)
        return Tensor(x - y * alpha)

    return vm_impl


@vm_impl_getters.register(P.Mul)
def vm_impl_mul(self):
    """Generate vm_impl function for Mul."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        return Tensor(x * y)

    return vm_impl


@vm_impl_getters.register(P.Conj)
def vm_impl_conj(self):
    """Generate vm_impl function for Conj."""

    def vm_impl(x):
        x = x.asnumpy()
        t = np.conj(x)
        return Tensor(t)

    return vm_impl


@vm_impl_getters.register(P.Square)
def vm_impl_square(self):
    """Generate vm_impl function for Square."""

    def vm_impl(x):
        x = x.asnumpy()
        return Tensor(x * x)

    return vm_impl


@vm_impl_getters.register(P.Sqrt)
def vm_impl_sqrt(self):
    """Generate vm_impl function for Sqrt."""

    def vm_impl(x):
        x = x.asnumpy()
        res = vm.sqrt(x)
        return Tensor(res)

    return vm_impl


@vm_impl_getters.register(P.Pow)
def vm_impl_pow(self):
    """Generate vm_impl function for Pow."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        res = vm.power(x, y)
        return Tensor(res)

    return vm_impl


@vm_impl_getters.register(P.Exp)
def vm_impl_exp(self):
    """Generate vm_impl function for Exp."""

    def vm_impl(x):
        x = x.asnumpy()
        res = vm.exp(x)
        return Tensor(res)

    return vm_impl


@vm_impl_getters.register(P.RealDiv)
def vm_impl_real_div(self):
    """Generate vm_impl function for RealDiv."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        out = x / y
        out = np.array(out, x.dtype)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Div)
def vm_impl_div(self):
    """Generate vm_impl function for Div."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        return Tensor(x / y)

    return vm_impl


@vm_impl_getters.register(P.ReduceMean)
def vm_impl_reduce_mean(self):
    """Generate vm_impl function for ReduceMean."""

    def vm_impl(x, axis, keep_dims):
        x = x.asnumpy()
        out = vm.mean(x, axis)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.ReduceMax)
def vm_impl_reduce_max(self):
    """Generate vm_impl function for ReduceMean."""

    def vm_impl(x, axis):
        x = x.asnumpy()
        if axis == ():
            axis = None
        out = np.amax(x, axis)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Equal)
def vm_impl_equal(self):
    """Generate vm_impl function for Equal."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        out = vm.equal(x, y)
        return Tensor(np.array(out))

    return vm_impl


@vm_impl_getters.register(P.NotEqual)
def vm_impl_not_equal(self):
    """Generate vm_impl function for NotEqual."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        out = vm.not_equal(x, y)
        return Tensor(np.array(out))

    return vm_impl


@vm_impl_getters.register("Greater")
def vm_impl_greater(self):
    """Generate vm_impl function for Greater."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        out = vm.greater(x, y)
        return Tensor(np.array(out))

    return vm_impl


@vm_impl_getters.register(P.Maximum)
def vm_impl_maximum(self):
    """Generate vm_impl function for Maximum."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        out = vm.maximum(x, y)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Minimum)
def vm_impl_minimum(self):
    """Generate vm_impl function for Minimum."""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        out = vm.minimum(x, y)
        return Tensor(out)

    return vm_impl


@vm_impl_getters.register(P.Less)
def vm_impl_less(self):
    """Generate vm_impl function for Less"""

    def vm_impl(x, y):
        x = x.asnumpy()
        y = y.asnumpy()
        out = vm.less(x, y)
        return Tensor(np.array(out))

    return vm_impl
