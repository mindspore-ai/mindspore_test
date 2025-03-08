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
from mindspore.ops.operations.other_ops import StopGradient
from mindspore.common.tensor import Tensor, _TensorMeta
from mindspore.common._stub_tensor import StubTensor

__all__ = ["Buffer"]


# Metaclass to combine _TensorMeta and the instance check override for Buffer.
class _BufferMeta(_TensorMeta):
    # Make `isinstance(t, Buffer)` return True for custom tensor instances that have the _is_buffer flag.
    def __instancecheck__(self, instance):  # pylint: disable=C0203
        if self is Buffer:
            if isinstance(instance, Tensor) and getattr(instance, "_is_buffer", False):
                return True
        return super().__instancecheck__(instance)


class Buffer(Tensor, metaclass=_BufferMeta):
    r"""A kind of Tensor that should not be considered a model
    parameter. For example, BatchNorm's ``running_mean`` is not a parameter, but is part of the Cell's state.

    Buffers are :class:`~mindspore.Tensor` subclasses, that have a
    very special property when used with :class:`Cell` s -- when they're
    assigned as Cell attributes they are automatically added to the list of
    its buffers, and will appear e.g. in :meth:`~mindspore.nn.Cell.buffers` iterator.
    Assigning a Tensor doesn't have such effect. One can still assign a Tensor as explicitly by using
    the :meth:`~mindspore.nn.Cell.register_buffer` function.

    Args:
        data (Tensor): buffer tensor.
        persistent (bool, optional): whether the buffer is part of the Cell's
            :attr:`state_dict`. Default: ``True``
    """

    def __new__(cls, data=None, *, persistent=True):
        if data is None:
            raise ValueError('For create Buffer, input data should not be None')
        if not isinstance(data, (Tensor, StubTensor)):
            raise TypeError('For create Buffer, type of input data should be Tensor')
        if isinstance(data, StubTensor):
            data = data.stub_sync()
        t = StopGradient()(data)
        t._is_buffer = True  # pylint: disable=W0212
        t.persistent = persistent
        return t
