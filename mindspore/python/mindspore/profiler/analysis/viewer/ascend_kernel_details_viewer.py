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

from mindspore import log as logger
from mindspore.profiler.analysis.viewer.base_viewer import BaseViewer
from mindspore.profiler.common.constant import OpSummaryHeaders
from mindspore.profiler.common.path_manager import PathManager


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
        except Exception as e: # pylint: disable=W0703
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

        if dev_kernels is None or not dev_kernels:
            logger.warning("device kernels is empty")
            return

        # build device kernel to framework launch op map
        dev_kernel_to_fwk_op = {}
        for _, per_tid_kernels in dev_kernels.items():
            for kernel in per_tid_kernels:
                dev_kernel_name = kernel.name
                dev_kerel_ts = str(kernel.ts)
                dev_kernel_to_fwk_op[(dev_kernel_name, dev_kerel_ts)] = kernel.parent.name

        launch_ops = [None] * len(self.op_summary)
        for index, summary in enumerate(self.op_summary):
            dev_kernel_name = summary[OpSummaryHeaders.OP_NAME.value]
            dev_kerel_ts = str(summary[OpSummaryHeaders.TASK_START_TIME.value]).strip("\t")
            fwk_lanch_op_name = dev_kernel_to_fwk_op.get((dev_kernel_name, dev_kerel_ts), None)

            if fwk_lanch_op_name is None:
                logger.warning(
                    "Can not find fwk launch op for dev kernel %s, ts %s",
                    dev_kernel_name,
                    dev_kerel_ts,
                )
                launch_ops[index] = dev_kernel_name
            else:
                launch_ops[index] = fwk_lanch_op_name

        # update op summary op name
        self.op_summary[OpSummaryHeaders.OP_NAME.value] = launch_ops
