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
from mindspore.profiler.analysis.parser.timeline_event.msprof_event import (
    MsprofCompleteEvent,
    MsprofInstantEvent,
    MsprofMetaEvent,
    MsprofCounterEvent
)


class TestMsprofCompleteEvent(unittest.TestCase):
    """Test cases for MsprofCompleteEvent class."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.event_data = {
            "name": "test_op",
            "ts": 1000,
            "dur": 500,
            "args": {"type": "op"}
        }
        self.event = MsprofCompleteEvent(self.event_data)

    def test_init_should_raise_type_error_when_data_not_dict(self):
        """Should raise TypeError when initialized with non-dict data."""
        invalid_inputs = [None, 42, "string", [], ()]
        for invalid_input in invalid_inputs:
            with self.assertRaises(TypeError) as context:
                MsprofCompleteEvent(invalid_input)
            self.assertEqual(str(context.exception), "Input data must be dict.")

    def test_properties_should_return_correct_values_when_initialized(self):
        """Should return correct values for all properties after initialization."""
        self.assertEqual(self.event.ts, Decimal('1000'))
        self.assertEqual(self.event.dur, Decimal('500'))
        self.assertEqual(self.event.name, "test_op")
        self.assertEqual(self.event.args, {"type": "op"})
        self.assertIsNone(self.event.parent)
        self.assertEqual(self.event.children, [])

    def test_parent_should_update_args_when_set(self):
        """Should update args with parent operation name when parent is set."""
        parent_data = {"name": "parent_op"}
        parent_event = MsprofCompleteEvent(parent_data)
        self.event.parent = parent_event
        self.assertEqual(self.event.parent, parent_event)
        self.assertEqual(self.event.args["mindspore_op"], "parent_op")

    def test_trace_format_should_return_correct_format_when_converted(self):
        """Should return correct Chrome trace format."""
        trace_format = self.event.to_trace_format()
        self.assertEqual(trace_format["ph"], EventConstant.COMPLETE_EVENT)
        self.assertEqual(trace_format["name"], "test_op")
        self.assertEqual(trace_format["ts"], "1000")
        self.assertEqual(trace_format["dur"], "500")
        self.assertEqual(trace_format["args"], {"type": "op"})


class TestMsprofInstantEvent(unittest.TestCase):
    """Test cases for MsprofInstantEvent class."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.event_data = {
            "name": "test_instant",
            "ts": 1000,
            "args": {"type": "instant"}
        }
        self.event = MsprofInstantEvent(self.event_data)

    def test_trace_format_should_return_correct_format_when_converted(self):
        """Should return correct Chrome trace format."""
        trace_format = self.event.to_trace_format()
        self.assertEqual(trace_format["ph"], EventConstant.INSTANT_EVENT)
        self.assertEqual(trace_format["name"], "test_instant")
        self.assertEqual(trace_format["ts"], "1000")
        self.assertEqual(trace_format["args"], {"type": "instant"})


class TestMsprofMetaEvent(unittest.TestCase):
    """Test cases for MsprofMetaEvent class."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.event_data = {
            "name": "test_meta",
            "args": {"type": "meta"}
        }
        self.event = MsprofMetaEvent(self.event_data)

    def test_trace_format_should_return_correct_format_when_converted(self):
        """Should return correct Chrome trace format."""
        trace_format = self.event.to_trace_format()
        self.assertEqual(trace_format["ph"], EventConstant.META_EVENT)
        self.assertEqual(trace_format["name"], "test_meta")
        self.assertEqual(trace_format["args"], {"type": "meta"})


class TestMsprofCounterEvent(unittest.TestCase):
    """Test cases for MsprofCounterEvent class."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.event_data = {
            "name": "test_counter",
            "ts": 1000,
            "args": {"type": "counter", "values": {"count": 1}}
        }
        self.event = MsprofCounterEvent(self.event_data)

    def test_trace_format_should_return_correct_format_when_converted(self):
        """Should return correct Chrome trace format."""
        trace_format = self.event.to_trace_format()
        self.assertEqual(trace_format["ph"], EventConstant.COUNTER_EVENT)
        self.assertEqual(trace_format["name"], "test_counter")
        self.assertEqual(trace_format["ts"], "1000")
        self.assertEqual(trace_format["args"], {"type": "counter", "values": {"count": 1}})


if __name__ == '__main__':
    unittest.main()
