# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
""" Mstx class for NPU profiling """
import mindspore
import mindspore._c_expression as c_expression

from mindspore import log as logging
from mindspore.runtime import Stream
from mindspore.profiler.common.constant import DeviceTarget


class Mstx:
    """
    Mstx class provides profiling tools for marking and tracing on NPU. This class provides three static methods: mark,
    range_start and range_end for adding marker points and ranges in profiling.
    """

    NPU_PROFILER = c_expression.Profiler.get_instance(DeviceTarget.NPU.value)

    @staticmethod
    def mark(message: str, stream: mindspore.runtime.Stream = None) -> None:
        """Add a marker point in profiling.

        Args:
            message (str): Description for the marker.
            stream (:class:`~.runtime.Stream`, optional): NPU stream for async execution, expected type:
                mindspore.runtime.Stream. Default: ``None``, which means only marking on host side without
                marking on device stream.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> import mindspore.dataset as ds
            >>> from mindspore import Profiler
            >>> from mindspore.profiler import ProfilerLevel, ProfilerActivity, schedule, tensor_board_trace_handler
            >>> from mindspore.profiler import mstx
            >>>
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.fc = nn.Dense(2,2)
            ...     def construct(self, x):
            ...         return self.fc(x)
            >>>
            >>> def generator():
            ...     for i in range(2):
            ...         yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))
            >>>
            >>> def train(net):
            ...     stream = ms.runtime.current_stream()
            ...     optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
            ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
            ...     data = ds.GeneratorDataset(generator, ["data", "label"])
            ...     model = ms.train.Model(net, loss, optimizer)
            ...     # Add marker before training
            ...     mstx.mark("train start", stream)
            ...     model.train(1, data)
            ...     # Add marker after training
            ...     mstx.mark("train end", stream)
            >>>
            >>> if __name__ == '__main__':
            ...     # Note: mstx only supports Ascend device and cannot be used in mindspore.nn.Cell.construct
            ...     # when in mindspore.GRAPH_MODE
            ...     ms.set_context(mode=ms.PYNATIVE_MODE)
            ...     ms.set_device(device_target="Ascend", device_id=0)
            ...     # Init Profiler
            ...     with Profiler(profiler_level=ProfilerLevel.LevelNone,
            ...                   on_trace_ready=tensor_board_trace_handler,
            ...                   activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            ...                   schedule=schedule(wait=0, warmup=0, active=3, repeat=1, skip_first=0),
            ...                   mstx=True) as profiler:
            ...         net = Net()
            ...         for i in range(5):
            ...             train(net)
            ...             profiler.step()
        """
        if not message or not isinstance(message, str):
            logging.warning("Invalid message for mstx.mark func. Please input valid message string.")
            return
        if stream:
            if isinstance(stream, Stream):
                device_stream = stream.device_stream()
                Mstx.NPU_PROFILER.mstx_mark(message, device_stream)
            else:
                logging.warning(
                    f"Invalid stream for mstx.mark func. Expected mindspore.runtime.Stream but got {type(stream)}.",
                )
        else:
            Mstx.NPU_PROFILER.mstx_mark(message)

    @staticmethod
    def range_start(message: str, stream: mindspore.runtime.Stream = None) -> int:
        """Start a profiling range.

        Args:
            message (str): Description for the range.
            stream (:class:`~.runtime.Stream`, optional): NPU stream for async execution, expected type:
                mindspore.runtime.Stream. Default: ``None``, which means only starting mstx range on
                host side without starting on device stream.

        Returns:
            int, range ID for range_end.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>> import mindspore.dataset as ds
            >>> from mindspore import Profiler
            >>> from mindspore.profiler import ProfilerLevel, ProfilerActivity, schedule, tensor_board_trace_handler
            >>> from mindspore.profiler import mstx
            >>>
            >>> class Net(nn.Cell):
            ...     def __init__(self):
            ...         super(Net, self).__init__()
            ...         self.fc = nn.Dense(2,2)
            ...     def construct(self, x):
            ...         return self.fc(x)
            >>>
            >>> def generator():
            ...     for i in range(2):
            ...         yield (np.ones([2, 2]).astype(np.float32), np.ones([2]).astype(np.int32))
            >>>
            >>> def train(net):
            ...     stream = ms.runtime.current_stream()
            ...     optimizer = nn.Momentum(net.trainable_params(), 1, 0.9)
            ...     loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
            ...     data = ds.GeneratorDataset(generator, ["data", "label"])
            ...     model = ms.train.Model(net, loss, optimizer)
            ...     # Start profiling range
            ...     range_id = mstx.range_start("training process", stream)
            ...     model.train(1, data)
            ...     # End profiling range
            ...     mstx.range_end(range_id)
            >>>
            >>> if __name__ == '__main__':
            ...     # Note: mstx only supports Ascend device and cannot be used in mindspore.nn.Cell.construct
            ...     # when in mindspore.GRAPH_MODE
            ...     ms.set_context(mode=ms.PYNATIVE_MODE)
            ...     ms.set_device(device_target="Ascend", device_id=0)
            ...     with Profiler(profiler_level=ProfilerLevel.LevelNone,
            ...                   on_trace_ready=tensor_board_trace_handler,
            ...                   activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            ...                   schedule=schedule(wait=0, warmup=0, active=3, repeat=1, skip_first=0),
            ...                   mstx=True) as profiler:
            ...         net = Net()
            ...         for i in range(5):
            ...             train(net)
            ...             profiler.step()
        """
        if not message or not isinstance(message, str):
            logging.warning("Invalid message for mstx.range_start func. Please input valid message string.")
            return 0
        # pylint: disable=no-else-return
        if stream:
            if isinstance(stream, Stream):
                device_stream = stream.device_stream()
                return Mstx.NPU_PROFILER.mstx_range_start(message, device_stream)
            else:
                logging.warning(
                    f"Invalid stream for mstx.range_start func. "
                    f"Expected mindspore.runtime.Stream but got {type(stream)}.",
                )
                return 0
        else:
            return Mstx.NPU_PROFILER.mstx_range_start(message)

    @staticmethod
    def range_end(range_id: int) -> None:
        """End a profiling range.

        Args:
            range_id (int): Range ID from range_start.

        Examples:
            >>> # Please refer to the example in range_start
            >>> # range_id = mstx.range_start("training process", stream)
            >>> # model.train(1, data)
            >>> # mstx.range_end(range_id)
        """
        if not isinstance(range_id, int):
            logging.warning(
                "Invalid message for mstx.range_start func. Please input return value from mstx.range_start."
            )
            return
        Mstx.NPU_PROFILER.mstx_range_end(range_id)
