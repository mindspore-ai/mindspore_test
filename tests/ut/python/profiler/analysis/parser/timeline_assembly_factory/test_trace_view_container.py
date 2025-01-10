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
from unittest.mock import Mock
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.analysis.parser.timeline_assembly_factory.trace_view_container import TraceViewContainer


# pylint: disable=protected-access
class TestTraceViewContainer(unittest.TestCase):
    """Test cases for TraceViewContainer class."""

    def setUp(self):
        """Set up test environment."""
        self.container = TraceViewContainer()

    def test_add_event_pool_should_store_pool_and_update_mappings(self):
        """Should store event pool and update name mappings."""
        pool = TimelineEventPool(pid=1)
        pool.name = "test_process"

        self.container.add_event_pool(pool)

        self.assertEqual(len(self.container.event_pools), 1)
        self.assertEqual(self.container.pid_to_name.get(1), "test_process")
        self.assertEqual(self.container.name_to_pid.get("test_process"), 1)

    def test_add_event_pool_should_raise_error_when_duplicate_name_exists(self):
        """Should raise ValueError for duplicate process names."""
        pool1 = TimelineEventPool(pid=1)
        pool1.name = "test_process"
        pool2 = TimelineEventPool(pid=2)
        pool2.name = "test_process"

        self.container.add_event_pool(pool1)

        with self.assertRaises(ValueError):
            self.container.add_event_pool(pool2)

    def test_get_pool_by_pid_should_return_pool_when_pid_exists(self):
        """Should return event pool for existing PID."""
        pool = TimelineEventPool(pid=1)
        self.container.event_pools[1] = pool

        result = self.container.get_pool_by_pid(1)

        self.assertEqual(result, pool)

    def test_get_pool_by_pid_should_return_none_when_pid_not_exists(self):
        """Should return None for non-existent PID."""
        result = self.container.get_pool_by_pid(999)

        self.assertIsNone(result)

    def test_get_pool_by_name_should_return_pool_when_name_exists(self):
        """Should return event pool for existing name."""
        pool = TimelineEventPool(pid=1)
        pool.name = "test_process"
        self.container.add_event_pool(pool)

        result = self.container.get_pool_by_name("test_process")

        self.assertEqual(result, pool)

    def test_get_pool_by_name_should_return_none_when_name_not_exists(self):
        """Should return None for non-existent process name."""
        result = self.container.get_pool_by_name("non_exist")

        self.assertIsNone(result)

    def test_add_trace_events_should_append_events_when_events_provided(self):
        """Should append trace events when events are provided."""
        events = [{"name": "event1"}, {"name": "event2"}]

        self.container.add_trace_events(events)

        self.assertEqual(self.container.trace_view, events)

    def test_get_trace_view_should_return_all_events_when_events_exist(self):
        """Should return all trace view events when events exist."""
        events = [{"name": "event1"}, {"name": "event2"}]
        self.container.trace_view = events

        result = self.container.get_trace_view()

        self.assertEqual(result, events)

    def test_get_all_pools_should_return_all_pools_when_pools_exist(self):
        """Should return list of all event pool instances when pools exist."""
        pool1 = TimelineEventPool(pid=1)
        pool2 = TimelineEventPool(pid=2)
        self.container.event_pools = {1: pool1, 2: pool2}

        pools = self.container.get_all_pools()

        self.assertEqual(len(pools), 2)
        self.assertIn(pool1, pools)
        self.assertIn(pool2, pools)

    def test_kernel_launch_op_event_should_store_and_retrieve_events_when_events_set(self):
        """Should properly store and retrieve kernel launch events when events are set."""
        # Create mock events with necessary attributes
        mock_event = Mock()
        mock_event.name = "kernel_launch_event"
        mock_event.ts = 1000
        mock_event.te = 2000
        mock_event.tid = 1

        events = {1: [mock_event]}

        self.container.kernel_launch_op_event = events

        # Verify the events are stored correctly
        self.assertEqual(self.container.kernel_launch_op_event, events)
        # Verify the mock event has the expected attributes
        stored_event = self.container.kernel_launch_op_event[1][0]
        self.assertEqual(stored_event.name, "kernel_launch_event")
        self.assertEqual(stored_event.ts, 1000)
        self.assertEqual(stored_event.te, 2000)
        self.assertEqual(stored_event.tid, 1)

    def test_hardware_op_event_should_store_and_retrieve_events_when_events_set(self):
        """Should properly store and retrieve hardware events when events are set."""
        # Create mock events with necessary attributes
        mock_event = Mock()
        mock_event.name = "hardware_event"
        mock_event.ts = 3000
        mock_event.te = 4000
        mock_event.tid = 2
        mock_event.parent = None
        mock_event.children = []

        events = {2: [mock_event]}

        self.container.hardware_op_event = events

        # Verify the events are stored correctly
        self.assertEqual(self.container.hardware_op_event, events)
        # Verify the mock event has the expected attributes
        stored_event = self.container.hardware_op_event[2][0]
        self.assertEqual(stored_event.name, "hardware_event")
        self.assertEqual(stored_event.ts, 3000)
        self.assertEqual(stored_event.te, 4000)
        self.assertEqual(stored_event.tid, 2)
        self.assertIsNone(stored_event.parent)
        self.assertEqual(stored_event.children, [])


if __name__ == '__main__':
    unittest.main()
