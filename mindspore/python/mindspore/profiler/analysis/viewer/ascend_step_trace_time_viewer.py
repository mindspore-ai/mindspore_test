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
"""Ascend Step Trace Time Viewer"""
import os
import re
from decimal import Decimal
from enum import Enum
from typing import List, Any, Tuple

import numpy as np

from mindspore import log as logger
from mindspore.profiler.analysis.viewer.base_viewer import BaseViewer
from mindspore.profiler.common.file_manager import FileManager
from mindspore.profiler.common.constant import TimelineLayerName, OverlapAnalysisTidName, ProfilerLevel
from mindspore.profiler.analysis.parser.timeline_event.msprof_event import (
    MsprofCompleteEvent,
)
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import (
    TimelineEventPool,
)
from mindspore.profiler.analysis.parser.timeline_assembly_factory.trace_view_container import (
    TraceViewContainer,
)
from mindspore.profiler.common.log import ProfilerLogger


class StepTraceTimeHeaders(Enum):
    """Step trace time headers"""
    STEP = "Step"
    COMPUTING = "Computing"
    COMMUNICATION_NOT_OVERLAPPED = "Communication(Not Overlapped)"
    OVERLAPPED = "Overlapped"
    COMMUNICATION = "Communication"
    FREE = "Free"
    STAGE = "Stage"
    BUBBLE = "Bubble"
    COMMUNICATION_NOT_OVERLAPPED_EXCLUDE_RECEIVE = "Communication(Not Overlapped and Exclude Receive)"
    PREPARING = "Preparing"


class AscendStepTraceTimeViewer(BaseViewer):
    """Ascend Step Trace Time Viewer"""

    STEP_TRACE_TIME_FILE_NAME = "step_trace_time.csv"
    STEP_TRACE_TIME_HEADERS = [header.value for header in StepTraceTimeHeaders]

    # HCCL Send, Recv op pattern
    PP_OP_PATTERN = (
        # eg: hcom_BatchSendRecv__101_0_1
        re.compile(r"hcom_\w+SendRecv__\d+"),
        # eg: hcom_send__101_0_1
        re.compile(r"hcom_send__\d+"),
        # eg: hcom_receive__101_0_1
        re.compile(r"hcom_receive__\d+"),
        re.compile(r"Receive-op"),
        re.compile(r"Send-op"),
    )

    # numpy array dtype
    OVERLAP_DTYPE = np.dtype([("ts", object), ("dur", object)])
    HCCL_DTYPE = np.dtype([("name", object), ("ts", object), ("dur", object)])

    def __init__(self, **kwargs):
        super().__init__()
        self._save_path = os.path.join(
            kwargs.get("ascend_profiler_output_path"), self.STEP_TRACE_TIME_FILE_NAME
        )
        self._profiler_level = kwargs.get("profiler_level")
        self._ascend_ms_dir = kwargs.get("ascend_ms_dir")
        ProfilerLogger.init(self._ascend_ms_dir)
        self._logger = ProfilerLogger.get_instance()
        self.step_trace_time_data = {}
        self.trace_container: TraceViewContainer = None
        self.hccl_pool: TimelineEventPool = None
        self.overlap_pool: TimelineEventPool = None
        # HCCL events
        self.hccl_events: List[MsprofCompleteEvent] = None
        # Overlap analysis events
        self.computing_events: List[MsprofCompleteEvent] = None
        self.communication_events: List[MsprofCompleteEvent] = None
        self.communication_not_overlapped_events: List[MsprofCompleteEvent] = None
        self.free_events: List[MsprofCompleteEvent] = None
        # Overlap analysis numpy array
        self.computing_np: np.ndarray = None
        self.communication_np: np.ndarray = None
        self.communication_not_overlapped_np: np.ndarray = None
        self.free_np: np.ndarray = None
        # HCCL numpy array
        self.hccl_events_np: np.ndarray = None

    def save(self, data: Any):
        """
        Save step trace time data to csv file
        """
        self._logger.info("AscendStepTraceTimeViewer start")
        if self._profiler_level == ProfilerLevel.LevelNone.value:
            return
        try:
            self._check_input_data(data)
            self._convert_events_to_numpy()
            self._calculate_step_trace_time()
            self._write_data()
        except Exception as e:  # pylint: disable=W0703
            self._logger.error("Failed to save step trace time data, error: %s", str(e), exc_info=True)
        self._logger.info("AscendStepTraceTimeViewer end")

    def _write_data(self):
        """
        Write step trace time data to csv file
        """
        self._logger.info("Write step trace time data start")
        data = [[str(self.step_trace_time_data.get(header, "")) for header in self.STEP_TRACE_TIME_HEADERS]]
        FileManager.create_csv_file(
            self._save_path,
            data,
            self.STEP_TRACE_TIME_HEADERS,
        )
        self._logger.info("Write step trace time data done, %d rows saved, save path: %s", len(data), self._save_path)

    def _check_input_data(self, data: Any):
        """
        Check input data and initialize data
        """
        self.trace_container: TraceViewContainer = data.get(
            "trace_view_container", None
        )

        if self.trace_container is None:
            raise ValueError("trace is empty")

        self.overlap_pool: TimelineEventPool = self.trace_container.get_pool_by_name(
            TimelineLayerName.OVERLAP_ANALYSIS.value
        )
        self.hccl_pool: TimelineEventPool = self.trace_container.get_pool_by_name(
            TimelineLayerName.HCCL.value
        )

        if self.overlap_pool is None:
            raise ValueError("overlap pool is empty")

        self.computing_events: List[MsprofCompleteEvent] = (
            self.overlap_pool.get_events_by_name(OverlapAnalysisTidName.COMPUTING.value)
        )
        self.communication_events: List[MsprofCompleteEvent] = (
            self.overlap_pool.get_events_by_name(
                OverlapAnalysisTidName.COMMUNICATION.value
            )
        )
        self.communication_not_overlapped_events: List[MsprofCompleteEvent] = (
            self.overlap_pool.get_events_by_name(
                OverlapAnalysisTidName.COMMUNICATION_NOT_OVERLAP.value
            )
        )
        self.free_events: List[MsprofCompleteEvent] = (
            self.overlap_pool.get_events_by_name(OverlapAnalysisTidName.FREE.value)
        )
        if self.hccl_pool is not None:
            self.hccl_events: List[MsprofCompleteEvent] = (
                self.hccl_pool.get_complete_events()
            )

    def _convert_overlap_events_to_numpy(
            self, events: List[MsprofCompleteEvent], dtype
    ):
        """
        Convert overlap events to numpy array
        """
        return np.array([(event.ts, event.dur) for event in events], dtype=dtype)

    def _convert_events_to_numpy(self):
        """
        Convert events to numpy array
        """
        self.computing_np = self._convert_overlap_events_to_numpy(
            self.computing_events, self.OVERLAP_DTYPE
        )
        self.communication_np = self._convert_overlap_events_to_numpy(
            self.communication_events, self.OVERLAP_DTYPE
        )
        self.communication_not_overlapped_np = self._convert_overlap_events_to_numpy(
            self.communication_not_overlapped_events, self.OVERLAP_DTYPE
        )
        self.free_np = self._convert_overlap_events_to_numpy(
            self.free_events, self.OVERLAP_DTYPE
        )
        self.computing_np = np.sort(self.computing_np, order="ts")
        self.communication_np = np.sort(self.communication_np, order="ts")
        self.communication_not_overlapped_np = np.sort(
            self.communication_not_overlapped_np, order="ts"
        )
        self.free_np = np.sort(self.free_np, order="ts")

        if self.hccl_events is not None:
            self.hccl_events_np = np.array(
                [(event.name, event.ts, event.dur) for event in self.hccl_events],
                dtype=self.HCCL_DTYPE,
            )
            self.hccl_events_np = np.sort(self.hccl_events_np, order="ts")

    def _calculate_step_trace_time(self):
        """
        Calculate step trace time data
        """
        self.step_trace_time_data[StepTraceTimeHeaders.STEP.value] = "0"
        self.step_trace_time_data[StepTraceTimeHeaders.COMPUTING.value] = np.sum(self.computing_np["dur"])
        self.step_trace_time_data[StepTraceTimeHeaders.COMMUNICATION.value] = np.sum(self.communication_np["dur"])
        self.step_trace_time_data[StepTraceTimeHeaders.COMMUNICATION_NOT_OVERLAPPED.value] = np.sum(
            self.communication_not_overlapped_np["dur"]
        )
        self.step_trace_time_data[StepTraceTimeHeaders.OVERLAPPED.value] = (
            self.step_trace_time_data[StepTraceTimeHeaders.COMMUNICATION.value]
            - self.step_trace_time_data[StepTraceTimeHeaders.COMMUNICATION_NOT_OVERLAPPED.value]
        )
        self.step_trace_time_data[StepTraceTimeHeaders.FREE.value] = np.sum(self.free_np["dur"])
        (
            self.step_trace_time_data[StepTraceTimeHeaders.STAGE.value],
            self.step_trace_time_data[StepTraceTimeHeaders.BUBBLE.value],
        ) = self._calculate_stage_bubble()

        self.step_trace_time_data[StepTraceTimeHeaders.COMMUNICATION_NOT_OVERLAPPED_EXCLUDE_RECEIVE.value] = (
            self.step_trace_time_data[StepTraceTimeHeaders.COMMUNICATION_NOT_OVERLAPPED.value]
            - self.step_trace_time_data[StepTraceTimeHeaders.BUBBLE.value]
        )
        self.step_trace_time_data[StepTraceTimeHeaders.PREPARING.value] = "0"

    def _calculate_stage_bubble(self) -> Tuple[Decimal, Decimal]:
        """
        Calculate stage and bubble time
        """
        if self.hccl_events is None:
            logger.info("HCCL events is empty, skip calculate stage and bubble")
            return Decimal(0), Decimal(0)

        total_hccl_time = self.hccl_events_np["ts"][-1] - self.hccl_events_np["ts"][0]
        bubble_time = np.sum(
            self.hccl_events_np["dur"][
                np.array(
                    [
                        self._is_send_recv_op(name)
                        for name in self.hccl_events_np["name"]
                    ]
                )
            ]
        )
        stage_time = total_hccl_time - bubble_time
        return stage_time, bubble_time

    def _is_send_recv_op(self, op_name: str) -> bool:
        """
        Check if the op is a send or recv op
        """
        return any(pattern.match(op_name) for pattern in self.PP_OP_PATTERN)
