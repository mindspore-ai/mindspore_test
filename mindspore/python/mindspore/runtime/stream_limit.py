# Copyright 2025 Huawei Technologies Co., Ltd
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
"""Runtime stream class"""
from mindspore._c_expression import get_stream_limit as get_stream_limit_
from mindspore._c_expression import set_stream_limit as set_stream_limit_
from mindspore._c_expression import reset_stream_limit as reset_stream_limit_
from .stream import Stream


def get_stream_limit(stream):
    r"""
    Return current stream limit core num.

    Note:
        This interface will synchronize the operator issuance, which may affect performance.

    Args:
        stream (Stream): selected stream.

    Returns:
        limit info (dict), stream limit core num.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> ms.runtime.get_stream_limit(ms.runtime.default_stream())
    """
    if not isinstance(stream, Stream):
        raise TypeError(
            f"For 'get_stream_limit', the argument 'stream' should be Stream,"
            f" but got {type(stream)}."
        )
    cube_num, vector_num = get_stream_limit_(stream)
    return {"cube_core_num": cube_num, "vector_core_num": vector_num}


def set_stream_limit(stream, cube_num=-1, vector_num=-1):
    r"""
    Sets the stream limit.

    Args:
        stream (Stream): set stream id.
        cube_num (int): set cube num for steam.
        vector_num (int): set vector num for steam.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> s1 = ms.runtime.Stream()
        >>> ms.runtime.set_stream_limit(s1, 8, 8)
    """
    if not isinstance(stream, Stream):
        raise TypeError(
            f"For 'set_stream_limit', the argument 'stream' should be Stream,"
            f" but got {type(stream)}."
        )
    set_stream_limit_(stream, cube_num, vector_num)


def reset_stream_limit(stream):
    r"""
    Reset the stream limit.

    Note:
        This interface will synchronize the operator issuance, which may affect performance.

    Args:
        stream (Stream): reset stream id.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> s1 = ms.runtime.Stream()
        >>> ms.runtime.reset_stream_limit(s1)
    """
    if not isinstance(stream, Stream):
        raise TypeError(
            f"For 'set_stream_limit', the argument 'stream' should be Stream,"
            f" but got {type(stream)}."
        )
    reset_stream_limit_(stream)


class StreamLimitCtx:
    r"""
    Context-manager that selects a given stream limit.

    All kernels queued within its context will be enqueued on a selected
    stream.

    Args:
        stream (Stream): selected stream.
        cube_num (int): set aic num for steam.
        vector_num (int): set aiv num for steam.

    Raises:
        TypeError: If 'stream' is neither a :class:`mindspore.runtime.Stream` nor a ``None``.

    Supported Platforms:
        ``Ascend``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> a = Tensor(np.ones([1024, 2048]), ms.float32)
        >>> b = Tensor(np.ones([2048, 4096]), ms.float32)
        >>> s1 = ms.runtime.Stream()
        >>> with ms.runtime.StreamLimitCtx(s1, 8, 8):
        ...     c = ops.matmul(a, b)
        >>> ms.runtime.synchronize()
    """
    def __init__(self, stream, cube_num=-1, vector_num=-1):
        if not isinstance(stream, Stream):
            raise TypeError(
                f"For 'StreamLimitCtx', the argument 'stream' should be Stream,"
                f" but got {type(stream)}."
            )
        self.stream = stream
        self.cube_num = cube_num
        self.vector_num = vector_num
        self.prev_cube_num = -1
        self.prev_vector_num = -1

    def __enter__(self):
        self.prev_cube_num, self.prev_vector_num = get_stream_limit_(self.stream)
        set_stream_limit_(self.stream, self.cube_num, self.vector_num)

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_stream_limit_(self.stream, self.prev_cube_num, self.prev_vector_num)
