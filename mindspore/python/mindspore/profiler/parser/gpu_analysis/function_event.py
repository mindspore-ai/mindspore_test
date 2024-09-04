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
"""Function event data struct."""
import struct

from mindspore.profiler.parser.ascend_analysis.constant import Constant
from mindspore.profiler.parser.ascend_analysis.function_event import MindSporeOpEnum, MindSporeOpEvent
from mindspore.profiler.parser.gpu_analysis.profiler_info_parser import GPUProfilerInfoParser


class GPUMindSporeOpEvent(MindSporeOpEvent):
    """
    Function event collected on the mindspore frame side.

    Args:
        data(Dict): The mindspore frame side data decoded by TLVDecoder.
    """

    def _init_params(self):
        """Initialize the attribute value of MindSporeOpEvent."""
        fix_size_data = struct.unpack(self._fix_data_format, self._orig_data.get(Constant.FIX_SIZE_BYTES))
        self.pid = Constant.MINDSPORE
        self.tid = int(fix_size_data[MindSporeOpEnum.START_THREAD_ID.value])
        self.name = str(self._orig_data.get(self._tlv_type_dict.get(Constant.OP_NAME), ""))
        self.ts = GPUProfilerInfoParser.get_local_time(fix_size_data[MindSporeOpEnum.START_NS.value])  # unit is us
        self.es = GPUProfilerInfoParser.get_local_time(fix_size_data[MindSporeOpEnum.END_NS.value])  # unit is us
        self.dur = self.es - self.ts
        self.flow_id = int(fix_size_data[MindSporeOpEnum.FLOW_ID.value])
        self.step = int(fix_size_data[MindSporeOpEnum.STEP_ID.value])
        self.level = int(fix_size_data[MindSporeOpEnum.LEVEL.value])
        self.custom_info = ""
        self.args = super()._get_args(fix_size_data)
