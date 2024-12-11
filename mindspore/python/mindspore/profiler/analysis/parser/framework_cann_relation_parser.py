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
"""Parser for analyzing the relationship between framework and CANN data."""
from typing import Dict, Any

from mindspore.profiler.analysis.parser.base_parser import BaseParser
from mindspore.profiler.analysis.parser.timeline_assembly_factory.ascend_timeline_assembler import (
    AscendTimelineAssembler
)


class FrameworkCannRelationParser(BaseParser):
    """FrameworkCannRelationParser"""

    def __init__(self, **kwargs):
        """Initialize the AscendTraceAnalyser."""
        super().__init__()
        self.assembler = AscendTimelineAssembler(**kwargs)

    def _parse(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the relation of framework and cann data."""
        self.assembler.assemble(data)
        trace_view_container = self.assembler.get_trace_view_container()
        data.update({
            "trace_view_container": trace_view_container,
        })
        return data
