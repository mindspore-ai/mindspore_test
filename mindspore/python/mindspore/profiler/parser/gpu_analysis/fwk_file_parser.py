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
"""MindSpore framework oprange file parser"""

import os
from collections import defaultdict

from mindspore import log as logger

from mindspore.profiler.parser.ascend_analysis.fwk_file_parser import FwkFileParser
from mindspore.profiler.parser.ascend_analysis.file_manager import FileManager
from mindspore.profiler.parser.ascend_analysis.tlv_decoder import TLVDecoder
from mindspore.profiler.parser.ascend_analysis.trace_event_manager import TraceEventManager
from mindspore.profiler.parser.ascend_analysis.constant import Constant

from mindspore.profiler.parser.gpu_analysis.function_event import GPUMindSporeOpEvent
from mindspore.profiler.parser.gpu_analysis.profiler_info_parser import GPUProfilerInfoParser
from mindspore.profiler.common.validator.validate_path import validate_and_normalize_path


class GPUFwkFileParser(FwkFileParser):
    """Framework-side operator file parser."""

    def __init__(self, source_path: str, rank_id: int):
        """
            source_path: The path of PROF_* directory
        """
        super(GPUFwkFileParser, self).__init__(source_path, rank_id)
        GPUProfilerInfoParser.init_source_path(source_path)
        GPUProfilerInfoParser.init_rank_id(rank_id)

    def get_op_range_data(self, step_list=None):
        """Read and decode all the mindspore oprange data."""
        op_range_list = []
        if os.path.exists(self._op_range_path):
            op_range_bytes = FileManager.read_file_content(self._op_range_path, "rb")
            op_range_list = TLVDecoder.decode(op_range_bytes, GPUMindSporeOpEvent, self._op_range_struct_size)
        else:
            logger.warning("Failed to find op_range data. skip parse host profiler data.")
        return op_range_list

    def get_fwk_trace_data(self, mindspore_op_data=None):
        """Generate chrome trace format json data from decoded oprange data."""
        if not mindspore_op_data:
            mindspore_op_data = self.get_op_range_data()
        tid_map = defaultdict(set)
        fwk_x_event_list = []
        dataset_op_data = []

        for mindspore_op in mindspore_op_data:
            if mindspore_op.name == Constant.FLOW_OP:
                continue

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

        return fwk_x_event_list + fwk_m_event_list

    def _init_framework_path(self, source_path: str):
        """Init the oprange data path."""
        source_path = validate_and_normalize_path(source_path)
        if not os.path.exists(source_path):
            raise FileNotFoundError("Input source_path does not exist!")
        self._prof_root = source_path
        self._op_range_path = os.path.join(source_path, self._op_range.format(self.rank_id))
