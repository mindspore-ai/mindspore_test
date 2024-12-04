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
""" MSTX class for NPU profiling """
import mindspore._c_expression as c_expression

from mindspore import log as logging
from mindspore.hal import Stream
from mindspore.profiler.common.constant import DeviceTarget


class Mstx:
    """MSTX class provides profiling tools for marking and tracing on NPU"""

    NPU_PROFILER = c_expression.Profiler.get_instance(DeviceTarget.NPU.value)

    @staticmethod
    def mark(message: str, stream: Stream = None) -> None:
        """Add a marker point in profiling

        Args:
            message: Description for the marker
            stream: NPU stream for async execution
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
                    f"Invalid stream for mstx.mark func. Expected mindspore.hal.Stream but got {type(stream)}.",
                )
        else:
            Mstx.NPU_PROFILER.mstx_mark(message)

    @staticmethod
    def range_start(message: str, stream: Stream = None) -> int:
        """Start a profiling range

        Args:
            message: Description for the range
            stream: NPU stream for async execution

        Returns:
            Range ID for range_end
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
                    f"Invalid stream for mstx.range_start func. Expected mindspore.hal.Stream but got {type(stream)}.",
                )
                return 0
        else:
            return Mstx.NPU_PROFILER.mstx_range_start(message)

    @staticmethod
    def range_end(range_id: int) -> None:
        """End a profiling range

        Args:
            range_id: Range ID from range_start
        """
        if not isinstance(range_id, int):
            logging.warning(
                "Invalid message for mstx.range_start func. Please input return value from mstx.range_start."
            )
            return
        Mstx.NPU_PROFILER.mstx_range_end(range_id)
