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

from decimal import Decimal
import unittest

from mindspore.profiler.common.constant import EventConstant
from mindspore.profiler.analysis.parser.timeline_event.flow_event import (
    FlowStartEvent,
    FlowEndEvent
)


class TestFlowStartEvent(unittest.TestCase):
    """Test cases for FlowStartEvent class."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.event_data = {
            "name": "test_flow",
            "id": "flow_1",
            "cat": "test_category",
            "ts": 1000,
            "pid": 1,
            "tid": 2
        }
        self.event = FlowStartEvent(self.event_data)

    def test_flow_properties_should_return_correct_values_when_initialized(self):
        """Should return correct values for flow-specific properties."""
        self.assertEqual(self.event.flow_id, "flow_1")
        self.assertEqual(self.event.cat, "test_category")
        self.assertEqual(self.event.ts, Decimal('1000'))
        self.assertEqual(self.event.ph, EventConstant.START_FLOW)

    def test_trace_format_should_return_complete_format_when_converted(self):
        """Should return complete Chrome trace format for flow event."""
        trace_format = self.event.to_trace_format()
        expected_format = {
            "name": "test_flow",
            "bp": "e",
            "ph": EventConstant.START_FLOW,
            "ts": "1000",
            "pid": 1,
            "tid": 2,
            "id": "flow_1",
            "cat": "test_category"
        }
        self.assertEqual(trace_format, expected_format)

    def test_flow_properties_should_return_defaults_when_data_missing(self):
        """Should return default values when flow data is missing."""
        empty_event = FlowStartEvent({})
        self.assertEqual(empty_event.flow_id, "")
        self.assertEqual(empty_event.cat, "")
        self.assertEqual(empty_event.ts, Decimal('0'))


class TestFlowEndEvent(unittest.TestCase):
    """Test cases for FlowEndEvent class."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.event_data = {
            "name": "test_flow",
            "id": "flow_1",
            "cat": "test_category",
            "ts": 2000,
            "pid": 1,
            "tid": 2
        }
        self.event = FlowEndEvent(self.event_data)

    def test_flow_properties_should_return_correct_values_when_initialized(self):
        """Should return correct values for flow-specific properties."""
        self.assertEqual(self.event.flow_id, "flow_1")
        self.assertEqual(self.event.cat, "test_category")
        self.assertEqual(self.event.ts, Decimal('2000'))
        self.assertEqual(self.event.ph, EventConstant.END_FLOW)

    def test_trace_format_should_return_complete_format_when_converted(self):
        """Should return complete Chrome trace format for flow event."""
        trace_format = self.event.to_trace_format()
        expected_format = {
            "name": "test_flow",
            "bp": "e",
            "ph": EventConstant.END_FLOW,
            "ts": "2000",
            "pid": 1,
            "tid": 2,
            "id": "flow_1",
            "cat": "test_category"
        }
        self.assertEqual(trace_format, expected_format)


if __name__ == '__main__':
    unittest.main()
