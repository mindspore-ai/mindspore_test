# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
"""Test the FrameworkCannRelationParser class."""
import unittest
from unittest.mock import patch, MagicMock

from mindspore.profiler.analysis.parser.framework_cann_relation_parser import FrameworkCannRelationParser
from mindspore.profiler.analysis.parser.timeline_assembly_factory.ascend_timeline_assembler import AscendTimelineAssembler
from mindspore.profiler.common.log import ProfilerLogger


# pylint: disable=protected-access
class TestFrameworkCannRelationParser(unittest.TestCase):
    @patch.object(ProfilerLogger, 'init')
    @patch.object(ProfilerLogger, 'get_instance')
    def setUp(self, mock_get_instance, mock_init):
        self.kwargs = {
            "ascend_ms_dir": "test_ascend_ms_dir"
        }
        mock_logger = unittest.mock.MagicMock()
        mock_get_instance.return_value = mock_logger
        mock_init.return_value = None
        self.parser = FrameworkCannRelationParser(**self.kwargs)

    @patch.object(AscendTimelineAssembler, 'assemble')
    @patch.object(AscendTimelineAssembler, 'get_trace_view_container')
    def test_parse_should_success_when_correct(self, mock_get_trace_view_container, mock_assemble):
        data = {}
        mock_trace_view_container = MagicMock()
        mock_get_trace_view_container.return_value = mock_trace_view_container
        result = self.parser._parse(data)

        mock_assemble.assert_called_once_with(data)
        mock_get_trace_view_container.assert_called_once()
        self.parser._logger.info.assert_any_call("FrameworkCannRelationParser assemble done")
        self.parser._logger.info.assert_any_call("FrameworkCannRelationParser get trace view container done")
        self.assertEqual(result["trace_view_container"], mock_trace_view_container)


if __name__ == "__main__":
    unittest.main()
