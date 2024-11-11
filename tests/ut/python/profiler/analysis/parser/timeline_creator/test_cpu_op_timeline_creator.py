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

import unittest
from unittest import mock

from mindspore.profiler.common.constant import EventConstant, TimelineLayerName
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.analysis.parser.timeline_creator.cpu_op_timeline_creator import CpuOpTimelineCreator


class TestCpuOpTimelineCreator(unittest.TestCase):
    """Test cases for CpuOpTimelineCreator."""

    def setUp(self):
        """Set up test environment."""
        self.creator = CpuOpTimelineCreator()
        self.normal_line = "Default/MatMul-op1;MatMul;1000,500,2"
        self.multi_time_line = "Default/Add-op3;Add;1500,200,2 2000,300,3"
        self.invalid_format_line = "Default/MatMul-op1;MatMul"  # Missing time info
        self.invalid_time_line = "Default/MatMul-op1;MatMul;1000,500"  # Invalid time format

    def test_create_should_create_pool_when_input_valid_lines(self):
        """Create event pool when input lines are valid."""
        self.creator.create([self.normal_line])

        self.assertIn(EventConstant.CPU_OP_PID, self.creator.event_pools)
        pool = self.creator.event_pools[EventConstant.CPU_OP_PID]
        events = pool.get_all_events()

        # 1 complete event + 3 process meta events + 2 thread meta events
        self.assertEqual(len(events), 6)

    def test_create_should_not_create_pool_when_input_empty_lines(self):
        """Not create event pool when input lines are empty."""
        self.creator.create([])
        self.assertEqual(len(self.creator.event_pools), 0)

    # pylint: disable=protected-access
    def test_create_base_events_should_create_events_when_input_single_time(self):
        """Create base events when input line has single time info."""
        pool = self.creator.event_pools.setdefault(
            EventConstant.CPU_OP_PID,
            TimelineEventPool(EventConstant.CPU_OP_PID)
        )
        self.creator._create_base_events(pool, [self.normal_line])

        complete_events = pool.get_complete_events()
        self.assertEqual(len(complete_events), 1)
        event = complete_events[0]
        self.assertEqual(event.name, "Default/MatMul-op1")
        self.assertEqual(event.tid, 2)
        self.assertEqual(event.args["type"], "MatMul")

    def test_create_base_events_should_create_events_when_input_multiple_times(self):
        """Create base events when input line has multiple time info."""
        pool = self.creator.event_pools.setdefault(
            EventConstant.CPU_OP_PID,
            TimelineEventPool(EventConstant.CPU_OP_PID)
        )
        self.creator._create_base_events(pool, [self.multi_time_line])

        complete_events = pool.get_complete_events()
        self.assertEqual(len(complete_events), 2)
        tids = {event.tid for event in complete_events}
        self.assertEqual(tids, {2, 3})

    def test_create_meta_events_should_create_process_and_thread_meta_events(self):
        """Create both process and thread meta events correctly."""
        pool = self.creator.event_pools.setdefault(
            EventConstant.CPU_OP_PID,
            TimelineEventPool(EventConstant.CPU_OP_PID)
        )
        # Add some events to create threads
        self.creator._create_base_events(pool, [self.multi_time_line])
        self.creator._create_meta_event(pool)

        meta_events = pool.get_all_events()

        # Check process meta events
        self.assertTrue(any(
            event.name == EventConstant.PROCESS_NAME and
            event.args["name"] == TimelineLayerName.CPU_OP.value
            for event in meta_events
        ))
        self.assertTrue(any(
            event.name == EventConstant.PROCESS_SORT and
            event.args["sort_index"] == EventConstant.CPU_OP_SORT_IDX
            for event in meta_events
        ))
        self.assertTrue(any(
            event.name == EventConstant.PROCESS_LABEL and
            event.args["labels"] == EventConstant.CPU_LABEL
            for event in meta_events
        ))

        # Check thread meta events
        thread_names = {
            event.args["name"] for event in meta_events
            if event.name == EventConstant.THREAD_NAME
        }
        self.assertEqual(thread_names, {"Thread 2", "Thread 3"})

    def test_create_should_log_warning_when_input_invalid_format(self):
        """Should log warning when input line has invalid format."""
        with mock.patch("mindspore.log.warning") as mock_warning:
            self.creator.create([self.invalid_format_line])
            mock_warning.assert_called_once_with(
                "Invalid CPU info format, expected at least 3 fields but got 2: "
                "Default/MatMul-op1;MatMul"
            )

    def test_create_should_log_warning_when_input_invalid_time_format(self):
        """Should log warning when input line has invalid time format."""
        with mock.patch("mindspore.log.warning") as mock_warning:
            self.creator.create([self.invalid_time_line])
            mock_warning.assert_called_once_with(
                "Invalid time info format, expected 3 fields but got 2: 1000,500"
            )

    def test_get_chrome_trace_data_should_return_formatted_events(self):
        """Return formatted events in Chrome trace format."""
        self.creator.create([self.normal_line])
        trace_events = self.creator.get_chrome_trace_data()
        self.assertTrue(any(event["name"] == "Default/MatMul-op1" for event in trace_events))
        self.assertTrue(any(event["name"] == EventConstant.PROCESS_NAME for event in trace_events))

    def test_get_chrome_trace_data_should_return_empty_when_no_pools(self):
        """Return empty list when no event pools exist."""
        trace_events = self.creator.get_chrome_trace_data()
        self.assertEqual(trace_events, [])

    def test_get_event_pools_should_return_all_pools(self):
        """Return all event pools."""
        self.creator.create([self.normal_line])
        pools = self.creator.get_event_pools()
        self.assertEqual(len(pools), 1)
        self.assertIn(EventConstant.CPU_OP_PID, pools)


if __name__ == '__main__':
    unittest.main()
