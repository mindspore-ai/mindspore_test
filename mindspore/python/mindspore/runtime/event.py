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
"""Runtime event class"""
from mindspore._c_expression import Event as Event_
from mindspore._c_expression import Stream as Stream_
from mindspore._c_expression import current_stream as current_stream_
from .device import _is_supported

function_event_status = {'Event': False, 'wait': False}


class Event(Event_):
    r"""
    Wrapper around a device event.

    Device events are synchronization markers that can be used to monitor the device’s progress,
    to accurately measure timing, and to synchronize device streams.

    The underlying device events are lazily initialized when the event is first recorded.

    Args:
        enable_timing (bool, optional): indicates if the event should measure time (default: ``False``)
        blocking (bool, optional): if ``True``, `wait` will be blocking (default: ``False``)

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> import mindspore as ms
        >>> import numpy as np
        >>> from mindspore import Tensor, ops
        >>> ms.set_device("Ascend", 0)
        >>> start = ms.runtime.Event(enable_timing=True)
        >>> end = ms.runtime.Event(enable_timing=True)
        >>> s1 = ms.runtime.Stream()
        >>> s2 = ms.runtime.Stream()
        >>> a = Tensor(np.ones([2, 2]), ms.float32)
        >>> b = Tensor(np.ones([2, 2]), ms.float32)
        >>> c = Tensor(np.ones([2, 2]), ms.float32)
        >>> with ms.runtime.StreamCtx(s1):
        ...     d = ops.matmul(a, b)
        ...     start.record()
        >>> c += 2
        >>> end.record()
        >>> with ms.runtime.StreamCtx(s2):
        ...     start.synchronize()
        ...     end.synchronize()
        ...     e = c + d
        >>> ms.runtime.synchronize()
        >>> print(e)
        [[5. 5.]
         [5. 5.]]
        >>> elapsed_time = start.elapsed_time(end)
    """

    def __init__(self, enable_timing=False, blocking=False):
        if not function_event_status['Event']:
            function_event_status['Event'] = True
            if not _is_supported():
                return
        if not isinstance(enable_timing, bool):
            raise TypeError(
                f"For 'Event', the 'enable_timing' must be a bool,"
                f" but got {type(enable_timing).__name__}."
            )
        if not isinstance(blocking, bool):
            raise TypeError(
                f"For 'Event', the 'blocking' must be a bool,"
                f" but got {type(blocking).__name__}."
            )
        super().__init__(enable_timing, blocking)

    def record(self, stream=None):
        r"""
        Records the event in a given stream.

        Uses :func:`mindspore.runtime.current_stream()` if no `stream` is specified. The
        stream's device must match the event's device.

        Args:
            stream (Stream, optional): a stream to record. If this argument is ``None``,
                current stream will be used. Default value: ``None``.

        Raises:
            TypeError: If 'stream' is neither a :class:`mindspore.runtime.Stream` nor a ``None``.
        """
        if stream is None:
            stream = current_stream_()
        if not isinstance(stream, Stream_):
            raise TypeError(
                f"For 'record', the argument 'stream' should be Stream,"
                f" but got {type(stream)}."
            )
        super().record(stream)

    def wait(self, stream=None):
        r"""
        Makes all future work submitted to the given stream wait for this
        event.

        Use :func:`mindspore.runtime.current_stream()` if no `stream` is specified.

        Args:
            stream (Stream, optional): a stream to record. If this argument is ``None``,
                current stream will be used. Default value: ``None``.

        Raises:
            TypeError: If 'stream' is neither a :class:`mindspore.runtime.Stream` nor a ``None``.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> from mindspore import Tensor, ops
            >>> ms.set_device("Ascend", 0)
            >>> event = ms.runtime.Event()
            >>> s1 = ms.runtime.Stream()
            >>> s2 = ms.runtime.Stream()
            >>> a = Tensor(np.ones([2, 2]), ms.float32)
            >>> b = Tensor(np.ones([2, 2]), ms.float32)
            >>> with ms.runtime.StreamCtx(s1):
            ...     c = ops.matmul(a, b)
            ...     event.record()
            >>> event.wait()
            >>> d = c + 2
            >>> ms.runtime.synchronize()
            >>> print(d)
            [[4. 4.]
             [4. 4.]]
        """
        if not function_event_status['wait']:
            function_event_status['wait'] = True
            if not _is_supported():
                return
        if stream is None:
            stream = current_stream_()
        if not isinstance(stream, Stream_):
            raise TypeError(
                f"For 'wait', the argument 'stream' should be Stream,"
                f" but got {type(stream)}."
            )
        super().wait(stream)

    def synchronize(self):
        r"""
        Waits for the event to complete.

        Waits until the completion of all work currently captured in this event.
        This prevents the CPU thread from proceeding until the event completes.
        """
        # pylint: disable=useless-super-delegation
        super().synchronize()

    def query(self):
        r"""
        Checks if all work currently captured by event has completed.

        Returns:
            A boolean indicating if all work currently captured by event has completed.

        Supported Platforms:
            ``Ascend`` ``GPU``

        Examples:
            >>> import mindspore as ms
            >>> import numpy as np
            >>> from mindspore import Tensor, ops
            >>> ms.set_device("Ascend", 0)
            >>> a = Tensor(np.ones([1024, 2048]), ms.float32)
            >>> b = Tensor(np.ones([2048, 4096]), ms.float32)
            >>> s1 = ms.runtime.Stream()
            >>> with ms.runtime.StreamCtx(s1):
            ...     c = ops.matmul(a, b)
            ...     ev = s1.record_event()
            >>> s1.synchronize()
            >>> assert ev.query()
        """
        # pylint: disable=useless-super-delegation
        return super().query()

    def elapsed_time(self, end_event):
        r"""
        Returns the time elapsed in milliseconds after the event was
        recorded and before the end_event was recorded.

        Args:
            end_event (Event): end event.

        Returns:
            float, the time elapsed in milliseconds.

        Raises:
            TypeError: If 'end_event' is not a :class:`mindspore.runtime.Event`.
        """
        # pylint: disable=useless-super-delegation
        if not isinstance(end_event, Event):
            raise TypeError(
                f"For 'elapsed_time', the argument 'end_event' should be Event,"
                f" but got {type(end_event)}."
            )
        return super().elapsed_time(end_event)
