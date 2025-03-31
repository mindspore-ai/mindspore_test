# Copyright 2020-2025 Huawei Technologies Co., Ltd
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

"""Buffer for cell."""
from mindspore.common.tensor import Tensor, _TensorMeta

__all__ = ["Buffer"]


# Metaclass to combine _TensorMeta and the instance check override for Buffer.
class _BufferMeta(_TensorMeta):
    # Make `isinstance(t, Buffer)` return True for custom tensor instances that have the _is_buffer flag.
    def __instancecheck__(cls, instance):
        if cls is Buffer:
            if isinstance(instance, Tensor) and getattr(instance, "_is_buffer", False):
                return True
        return super().__instancecheck__(instance)


class Buffer(Tensor, metaclass=_BufferMeta):
    r"""A kind of Tensor that should not be considered a model
    parameter. For example, BatchNorm's `running_mean` is not a parameter, but is part of the Cell's state.

    Buffers are :class:`~mindspore.Tensor` subclasses, that have a
    very special property when used with :class:`~.nn.Cell` s: when they're
    assigned as Cell attributes they are automatically added to the list of
    its buffers, and will appear e.g. in :func:`mindspore.nn.Cell.buffers` iterator.
    Assigning a tensor doesn't have such effect. One can still assign a tensor as a buffer explicitly by using
    the :func:`mindspore.nn.Cell.register_buffer` function.

    Args:
        data (Tensor): buffer tensor. Default ``None``.

    Keyword Args:
        persistent (bool, optional): whether the buffer is part of the Cell's
            :attr:`state_dict`. Default ``True``.
    """

    def __new__(cls, data=None, *, persistent=True):
        if data is None:
            raise ValueError('For create Buffer, input data should not be None')
        if not isinstance(data, Tensor):
            raise TypeError('For create Buffer, type of input data should be Tensor')
        from mindspore.ops import stop_gradient
        t = stop_gradient(data)
        t._is_buffer = True  # pylint: disable=W0212
        t.persistent = persistent
        return t
