# Copyright 2022 Huawei Technologies Co., Ltd
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

"""Defines parameter operators with functional form."""

from mindspore.ops import operations as P
from mindspore.ops._primitive_cache import _get_cache_prim
from mindspore.ops.auto_generate import assign, assign_add, assign_sub


def index_add(x, indices, y, axis, use_lock=True, check_index_bound=True):
    """
    Adds tensor `y` to specified axis and indices of `x`. The axis should be in [0,  len(x.dim) - 1],
    and indices should be in [0, x.shape[axis] - 1] at the axis dimension.

    Args:
        x (Union[Parameter, Tensor]): The input Parameter or Tensor to add to.
        indices (Tensor): Add the value of `x` and `y` along the dimension of the `axis` according to the
            specified index value, with data type int32.
            The `indices` must be 1D with the same size as the size of `y` in the `axis` dimension. The values
            of `indices` should be in [0, b), where the b is the size of `x` in the `axis` dimension.
        y (Tensor): The input tensor with the value to add. Must have same data type as `x`.
            The shape must be the same as `x` except the `axis` th dimension.
        axis (int): The dimension along which to index.
        use_lock (bool, optional): Whether to enable a lock to protect the updating process of variable tensors.
            If ``True`` , when updating the value of `x`, this process will be protected by a lock by using atomic
            operation.
            If ``False`` , the result may be unpredictable. Default: ``True`` .
        check_index_bound (bool, optional): If ``True``, check index boundary. If ``False`` ,
            don't check index boundary. Default: ``True`` .

    Returns:
        Tensor, has the same shape and dtype as `x`.

    Raises:
        TypeError: If neither `indices` nor `y` is a Tensor.
        ValueError: If axis is out of `x` rank's range.
        ValueError: If `x` rank is not the same as `y` rank.
        ValueError: If shape of `indices` is not 1D or size of `indices` is not equal to dimension of y[axis].
        ValueError: If `y`'s shape is not the same as `x` except the `axis` th dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore
        >>> from mindspore import Tensor, Parameter
        >>> from mindspore import ops
        >>> x = Parameter(Tensor(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), mindspore.float32), name="name_x")
        >>> indices = Tensor(np.array([0, 2]), mindspore.int32)
        >>> y = Tensor(np.array([[0.5, 1.0], [1.0, 1.5], [2.0, 2.5]]), mindspore.float32)
        >>> output = ops.index_add(x, indices, y, 1)
        >>> print(output)
        [[ 1.5  2.   4. ]
         [ 5.   5.   7.5]
         [ 9.   8.  11.5]]
    """
    _index_add = _get_cache_prim(P.IndexAdd)(axis, use_lock, check_index_bound)
    return _index_add(x, indices, y)


__all__ = [
    'assign',
    'assign_sub',
    'assign_add',
    'index_add'
]
__all__.sort()
