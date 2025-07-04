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

"""Implementation for internal polymorphism `zeros_like` operations."""

from __future__ import absolute_import
from mindspore.ops.composite import base
from mindspore.ops import functional as F
from mindspore.ops.operations import _sequence_ops as seq
from mindspore.ops.primitive import Primitive

zeros_like_leaf = base.MultitypeFuncGraph('zeros_like', True)
"""
`zeros_like` is a metafuncgraph object which will generate a tensor filled with one according to its input type
using ".register" decorator.
"""


@zeros_like_leaf.register("Number")
def _zeros_like_scalar(x):
    """Returns 0 which has the same dtype as x where x is a scalar."""
    if isinstance(x, int):
        return 0
    return 0.


@zeros_like_leaf.register("Bool")
def _zeros_like_bool(x):
    """Returns False if x is a bool."""
    return False


@zeros_like_leaf.register("String")
def _zeros_like_string(x):
    """Returns False if x is a string."""
    return 0


@zeros_like_leaf.register("Tuple")
def _zeros_like_tuple(x):
    """Returns False if x is a tuple."""
    return 0


@zeros_like_leaf.register("Function")
def _zeros_like_func(x):
    """
    Derivation of a function.

    Args:
        x (Function): x

    Returns:
        A instance of EnvType.
    """
    # Unused parameters are placeholders.
    return F.environ_create()


@zeros_like_leaf.register("Tensor")
def _zeros_like_tensor(x):
    """Returns a tensor with the same shape and dtype as x and all elements are 0."""
    return F.zeros_like(x)


@zeros_like_leaf.register("COOTensor")
def _zeros_like_coo_tensor(x):
    """Returns a tensor with the same shape and dtype as x and all elements are 0."""
    values = F.zeros_like(x.values)
    return F.make_coo_tensor(x.indices, values, x.shape)


@zeros_like_leaf.register("CSRTensor")
def _zeros_like_csr_tensor(x):
    """Returns a tensor with the same shape and dtype as x and all elements are 0."""
    values = F.zeros_like(x.values)
    return F.make_csr_tensor(x.indptr, x.indices, values, x.shape)


@zeros_like_leaf.register("MapTensor")
def _zeros_like_map_tensor(x):
    """Returns a map tensor with the same shape and dtype as x and all elements are 0."""
    return x


@zeros_like_leaf.register("TypeType")
def _zeros_like_type_type(x):
    """Returns x because x is a type. This is usually used in backprop progress."""
    return x


@zeros_like_leaf.register("None")
def _zeros_like_type_none(x):
    """Returns None where x is and should be None. This is usually used in backprop progress."""
    return x


@zeros_like_leaf.register("RefKeyType")
def _zeros_like_refkey_type(x):
    """
    Derivation of a type.

    Args:
        x (RefKeyType): x

    Returns:
        RefKeyType.
    """
    return x


@zeros_like_leaf.register("Problem")
def _zeros_like_abstract_error(x):
    """
    Derivation of a AbstractError.

    Args:
        x (AbstractError): return x

    Returns:
        x.
    """
    return x


@zeros_like_leaf.register("Dictionary")
def _zeros_like_dict(x):
    """
    Derivation of a AbstractError.

    Args:
        x (dict): the input

    Returns:
        dict, keys are same as input's keys, and value are same as zeros_like of input'value.
    """
    keys = x.keys()
    values = x.values()
    new_values = ()
    for ele in values:
        new_values += (zeros_like_leaf(ele),)
    return F.make_dict(keys, new_values)


_extract_keyword_arg = Primitive("extract_keyword_arg")


@zeros_like_leaf.register("Keyword")
def _zeros_like_keyword(x):
    """
    Handle Keyword input.

    Args:
        x (Keyword): the input

    Returns:
        zeros_like_leaf.
    """
    return zeros_like_leaf(_extract_keyword_arg(x))


@zeros_like_leaf.register("UMonad")
def _zeros_like_u_monad(x):
    """
    U Monad.

    Args:
        x (UMonad):

    Returns:
        x.
    """
    return x


@zeros_like_leaf.register("IOMonad")
def _zeros_like_io_monad(x):
    """
    IO Monad.

    Args:
        x (IOMonad):

    Returns:
        x.
    """
    return x


@zeros_like_leaf.register("EnvType")
def _zeros_like_env_type(x):
    """
    Env Type.

    Args:
        x (EnvType): the input

    Returns:
        a EnvType created by F.environ_create.
    """
    return F.environ_create()


@zeros_like_leaf.register("Slice")
def _zeros_like_slice(x):
    """
    Slice.

    Args:
        x (Slice): the input

    Returns:
        x.
    """
    return x


class _ZerosLike(base.ZerosLike_):
    def __init__(self, name, fn_leaf):
        """Initialize _ZerosLike."""
        base.ZerosLike_.__init__(self, name, fn_leaf)
        self.fn_leaf = fn_leaf

    def __call__(self, x):
        if isinstance(x, (tuple, list)) and F.is_sequence_shape_unknown(x):
            return seq.SequenceZerosLike()(x)
        return self.fn_leaf(x)


zeros_like = _ZerosLike('zeros_like', zeros_like_leaf)
