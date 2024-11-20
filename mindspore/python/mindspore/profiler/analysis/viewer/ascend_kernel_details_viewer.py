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
"""Ascend kernel details viewer"""
import os
import csv
from decimal import Decimal

from mindspore import log as logger
from mindspore.profiler.analysis.viewer.base_viewer import BaseViewer
from mindspore.profiler.analysis.parser.timeline_assembly_factory.trace_view_container import TraceViewContainer
from mindspore.profiler.common.constant import OpSummaryHeaders
from mindspore.profiler.common.path_manager import PathManager
from mindspore.profiler.common.constant import TimelineLayerName
from mindspore.profiler.common.constant import ProfilerStepNameConstant, JitLevel


class AscendKernelDetailsViewer(BaseViewer):
    """
    Ascend kernel details viewer
    """
    KERNEL_DETAILS_FILE_NAME = "kernel_details.csv"
    EXCLUDE_HEADERS = [OpSummaryHeaders.DEVICE_ID.value]
    RENAME_HEADERS = {
        OpSummaryHeaders.OP_NAME.value: "Name",
        OpSummaryHeaders.OP_TYPE.value: "Type",
        OpSummaryHeaders.TASK_TYPE.value: "Accelerator Core",
        OpSummaryHeaders.TASK_START_TIME.value: "Start Time(us)",
        OpSummaryHeaders.TASK_DURATION.value: "Duration(us)",
        OpSummaryHeaders.TASK_WAIT_TIME.value: "Wait Time(us)",
    }

    def __init__(self, **kwargs):
        super().__init__()
        self._save_path = os.path.join(
            kwargs.get("ascend_profiler_output_path"),
            self.KERNEL_DETAILS_FILE_NAME
        )
        self._is_set_schedule = kwargs.get("is_set_schedule")
        self._jit_level = kwargs.get("jit_level")
        self.op_summary_headers = None
        self.op_summary = None
        self.trace_container = None
        self.kernel_details_headers = None

    def save(self, data):
        """
        Save kernel details to csv file.
        """
        try:
            self._check_input_data(data)
            self._update_kernel_name()
            self._update_headers()
            self._write_data()
        except Exception as e:  # pylint: disable=W0703
            logger.warning("Failed to save kernel details: %s", str(e))

    def _check_input_data(self, data):
        """
        Check input data.
        """
        self.trace_container = data.get("trace_view_container", None)
        self.op_summary = data.get("op_summary", None)
        self.op_summary_headers = data.get("op_summary_headers", None)

        if self.op_summary is None or self.op_summary.size == 0:
            raise ValueError("op summary is empty")

        if self.trace_container is None:
            raise ValueError("trace view container is None")

    def _write_data(self):
        """
        Write data to csv file.
        """
        PathManager.check_directory_path_writeable(os.path.dirname(self._save_path))
        with open(self._save_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(self.kernel_details_headers)
            for row in self.op_summary:
                writer.writerow([row[field] for field in self.op_summary_headers])

    def _update_headers(self):
        """
        Update kernel details headers.
        """
        # filter exclude headers
        self.op_summary_headers = [
            header
            for header in self.op_summary_headers
            if header not in self.EXCLUDE_HEADERS
        ]
        if not self._is_set_schedule or self._jit_level == JitLevel.GRAPH_LEVEL:
            self.op_summary_headers.remove(OpSummaryHeaders.STEP_ID.value)
        # rename headers
        self.kernel_details_headers = [
            self.RENAME_HEADERS.get(header, header)
            for header in self.op_summary_headers
        ]

    def _update_kernel_name(self):
        """
        Update op summary op name to framework launch op name.
        """
        dev_kernels = self.trace_container.hardware_op_event
        _generate_step_id_by_trace_container(self.trace_container)

        if dev_kernels is None or not dev_kernels:
            logger.warning("device kernels is empty")
            return

        # build device kernel to framework launch op map
        dev_kernel_to_fwk_op = {}
        for _, per_tid_kernels in dev_kernels.items():
            for kernel in per_tid_kernels:
                dev_kernel_name = kernel.name
                dev_kerel_ts = str(kernel.ts)
                dev_kernel_to_fwk_op[(dev_kernel_name, dev_kerel_ts)] = kernel

        launch_ops = [None] * len(self.op_summary)
        step_ids = [None] * len(self.op_summary)
        for index, summary in enumerate(self.op_summary):
            dev_kernel_name = summary[OpSummaryHeaders.OP_NAME.value]
            dev_kernel_ts = str(summary[OpSummaryHeaders.TASK_START_TIME.value]).strip("\t")
            fwk_langch_op_name = None
            step_id = None
            if dev_kernel_to_fwk_op.get((dev_kernel_name, dev_kernel_ts)):
                kernel = dev_kernel_to_fwk_op.get((dev_kernel_name, dev_kernel_ts))
                if kernel.parent:
                    fwk_langch_op_name = kernel.parent.name
                step_id = kernel.step_id

            if fwk_langch_op_name is None:
                logger.warning(
                    "Can not find fwk launch op for dev kernel %s, ts %s",
                    dev_kernel_name,
                    dev_kernel_ts,
                )
                launch_ops[index] = dev_kernel_name
            else:
                launch_ops[index] = f"{fwk_langch_op_name}/{dev_kernel_name}"

            if step_id is None and self._is_set_schedule:
                logger.warning(
                    "Can not find step id for dev kernel %s, ts %s",
                    dev_kernel_name,
                    dev_kernel_ts,
                )
            else:
                step_ids[index] = step_id

        # update op summary op name
        self.op_summary[OpSummaryHeaders.OP_NAME.value] = launch_ops

        # update op summary step id
        if self._is_set_schedule:
            if not self._jit_level == JitLevel.GRAPH_LEVEL:
                self.op_summary[OpSummaryHeaders.STEP_ID.value] = step_ids


def _generate_step_id_by_trace_container(trace_container: TraceViewContainer):
    """
    Generate the step id of the kernel event.
    """
    # Retrieve all events from the trace container for the Mindspore timeline layer
    events = trace_container.get_pool_by_name(TimelineLayerName.MINDSPORE.value).get_all_events()

    # Filter events that contain "ProfilerStep" and create a dictionary mapping (start_ts, end_ts) to step ID
    step_time_to_id_dict = {}
    for event in events:
        event_name = event.name
        if ProfilerStepNameConstant.PROFILER_STEP in event_name:
            step_time_to_id_dict[event.name.split("#")[-1]] = (event.ts, event.dur + event.ts)

    # Get kernel launch events and sort them by timestamp
    hardware_op_events_dict = trace_container.hardware_op_event
    for hardware_op_events_list in hardware_op_events_dict.values():
        # Associate each hardware operation event with its step ID
        for hardware_op_event in hardware_op_events_list:
            kernel_event = hardware_op_event.parent
            if not kernel_event:
                continue

            hardware_op_event.step_id = _get_step_id_by_ts(kernel_event.ts, step_time_to_id_dict)


def _get_step_id_by_ts(ts: Decimal, step_events_dict: dict):
    """
    Retrieves the step ID for a given timestamp from the step events dictionary.

    Args:
        ts (Decimal): The timestamp for which to find the step ID.
        step_events_dict (dict): A dictionary mapping (start_ts, end_ts) to step ID.

    Returns:
        str: The step ID corresponding to the given timestamp, or None if not found.

    Raises:
        None
    """
    # Iterate through the step events dictionary to find the step ID for the given timestamp
    for step_id, (st, et) in step_events_dict.items():
        if st <= ts <= et:
            return step_id

    return None
