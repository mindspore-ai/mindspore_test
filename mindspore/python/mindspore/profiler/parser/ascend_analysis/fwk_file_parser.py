# Copyright 2023 Huawei Technologies Co., Ltd
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
"""MindSpore framework oprange file parser"""

import os
import stat
import csv
from typing import List
from collections import defaultdict

from mindspore import log as logger
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path
from mindspore.profiler.parser.ascend_analysis.file_manager import FileManager
from mindspore.profiler.parser.ascend_analysis.tlv_decoder import TLVDecoder
from mindspore.profiler.parser.ascend_analysis.function_event import MindSporeOpEvent
from mindspore.profiler.parser.ascend_analysis.trace_event_manager import TraceEventManager
from mindspore.profiler.parser.ascend_analysis.constant import Constant


class FwkFileParser:
    """Framework-side operator file parser."""

    _op_range = "FRAMEWORK/op_range_{}"
    _op_range_struct_size = 74

    def __init__(self, source_path: str, rank_id: int):
        """
            source_path: The path of PROF_* directory
        """
        self.rank_id = rank_id
        self._prof_root = None
        self._op_range_path = None
        self._init_framework_path(source_path)

    def get_op_range_data(self, step_list=None) -> List[MindSporeOpEvent]:
        """Read and decode all the mindspore oprange data."""
        op_range_list = []
        if os.path.exists(self._op_range_path):
            op_range_bytes = FileManager.read_file_content(self._op_range_path, "rb")
            op_range_list = TLVDecoder.decode(op_range_bytes, MindSporeOpEvent, self._op_range_struct_size)
        else:
            logger.warning("Failed to find op_range data. skip parse host profiler data.")
        if step_list and isinstance(step_list, list):
            first_step = min(op.step for op in op_range_list)
            step_list = [step - 1 + first_step for step in step_list]
            op_range_list = list(filter(lambda op: op.step in step_list, op_range_list))
        return op_range_list

    def get_fwk_trace_data(self, mindspore_op_data: List[MindSporeOpEvent] = None):
        """Generate chrome trace format json data from decoded oprange data."""
        if not mindspore_op_data:
            mindspore_op_data = self.get_op_range_data()
        tid_map = defaultdict(set)
        fwk_x_event_list = []
        link_event_dict = defaultdict(list)
        fwk_launch_op = defaultdict(list)
        dataset_op_data = []

        for mindspore_op in mindspore_op_data:
            if mindspore_op.name == Constant.FLOW_OP:
                link_event_dict[mindspore_op.flow_id].insert(0, mindspore_op)
                continue
            if mindspore_op.flow_id != Constant.INVALID_FLOW_ID:
                link_event_dict[mindspore_op.flow_id].append(mindspore_op)

            if 'KernelLaunch' in mindspore_op.name or 'LaunchTask' in mindspore_op.name:
                fwk_launch_op[mindspore_op.tid].append(mindspore_op)

            if mindspore_op.name.split('::')[0] == 'Dataset':
                dataset_op_data.append(mindspore_op)

            tid_map[mindspore_op.pid].add(mindspore_op.tid)
            if mindspore_op.dur > 0:
                fwk_x_event_list.append(TraceEventManager.create_x_event(mindspore_op, "cpu_op"))
            else:
                fwk_x_event_list.append(TraceEventManager.create_i_event(mindspore_op))

        fwk_m_event_list = []
        for pid, tid_set in tid_map.items():
            fwk_m_event_list.extend(TraceEventManager.create_m_event(pid, tid_set, pid))

        self.calculate_dataset_item(dataset_op_data)

        return fwk_x_event_list + fwk_m_event_list, link_event_dict, fwk_launch_op

    def calculate_dataset_item(self, dataset_op_data: List[MindSporeOpEvent]):
        """
        Get the summary data of dataset op by parsing the dataset_op_data.
        """
        self._prof_root = validate_and_normalize_path(self._prof_root)
        dataset_file = os.path.join(self._prof_root, f'dataset_{self.rank_id}.csv')
        summary_data = defaultdict(list)
        for op_data in dataset_op_data:
            _, event, stage = op_data.name.split('::')
            key = event + '::' + stage + '::' + op_data.custom_info
            summary_data[key].append(op_data.dur)

        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        modes = stat.S_IWUSR | stat.S_IRUSR
        with os.fdopen(os.open(dataset_file, flags, modes), 'w', newline='') as fw:
            csv_writer = csv.writer(fw)
            csv_writer.writerow(['Operation', 'Stage', 'Occurrences', 'Avg. time (us)', 'Custom Info'])
            for k, v in summary_data.items():
                event, stage, custom_info = k.split('::')
                count = len(v)
                average_execution = round(float(sum(v) / count), 2)
                csv_writer.writerow([event, stage, count, average_execution, custom_info])
        os.chmod(dataset_file, modes)

    def _init_framework_path(self, source_path: str):
        """Init the oprange data path."""
        source_path = validate_and_normalize_path(source_path)
        if not os.path.exists(source_path):
            raise FileNotFoundError("Input source_path does not exist!")
        device_name = os.path.basename(source_path)
        if not device_name.startswith("device") and not os.path.isdir(source_path):
            raise RuntimeError("Input source_path is invalid!")
        self._prof_root = os.path.dirname(os.path.dirname(source_path))
        self._op_range_path = os.path.join(self._prof_root, self._op_range.format(self.rank_id))
