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
from mindspore.profiler.analysis.parser.timeline_event.cpu_op_event import CpuOpCompleteEvent, CpuOpMetaEvent


class TestCpuOpCompleteEvent(unittest.TestCase):
    """Test cases for CpuOpCompleteEvent class."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.event_data = {
            "name": "test_cpu_op",
            "ts": 1000000,  # ns
            "dur": 1,  # ms
            "args": {"type": "cpu_op"}
        }
        self.event = CpuOpCompleteEvent(self.event_data)

    def test_properties_should_return_correct_values_when_initialized(self):
        """Should return correct values for all properties after initialization."""
        # Test timestamp conversions
        self.assertEqual(self.event.ts, Decimal('1000.000'))  # ns converted to us
        self.assertEqual(self.event.dur, Decimal('1000.000'))  # ms converted to us
        self.assertEqual(self.event.pid, int(EventConstant.CPU_OP_PID))

    def test_trace_format_should_return_correct_format_when_converted(self):
        """Should return correct Chrome trace format."""
        trace_format = self.event.to_trace_format()
        self.assertEqual(trace_format["ph"], EventConstant.COMPLETE_EVENT)
        self.assertEqual(trace_format["name"], "test_cpu_op")
        self.assertEqual(trace_format["ts"], "1000.000")
        self.assertEqual(trace_format["dur"], "1000.000")
        self.assertEqual(trace_format["pid"], int(EventConstant.CPU_OP_PID))


class TestCpuOpMetaEvent(unittest.TestCase):
    """Test cases for CpuOpMetaEvent class."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.event_data = {
            "name": "test_meta",
            "args": {"type": "meta"}
        }
        self.event = CpuOpMetaEvent(self.event_data)

    def test_properties_should_return_correct_values_when_initialized(self):
        """Should return correct values for all properties after initialization."""
        self.assertEqual(self.event.pid, int(EventConstant.CPU_OP_PID))
        self.assertEqual(self.event.name, "test_meta")
        self.assertEqual(self.event.args, {"type": "meta"})

    def test_trace_format_should_return_correct_format_when_converted(self):
        """Should return correct Chrome trace format."""
        trace_format = self.event.to_trace_format()
        self.assertEqual(trace_format["ph"], EventConstant.META_EVENT)


if __name__ == '__main__':
    unittest.main()
