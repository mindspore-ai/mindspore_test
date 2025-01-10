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

from mindspore.profiler.common.constant import EventConstant
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.analysis.parser.timeline_event.base_event import BaseEvent, MetaEvent


# pylint: disable=protected-access
class TestTimelineEventPool(unittest.TestCase):
    """Test cases for TimelineEventPool."""

    def setUp(self):
        """Set up test environment."""
        self.pid = 1
        self.pool = TimelineEventPool(pid=self.pid)

    def test_add_event_should_store_complete_event_when_phase_is_complete(self):
        """Store complete event when phase is complete."""
        event = mock.Mock(ph=EventConstant.COMPLETE_EVENT, tid=1)

        self.pool.add_event(event)

        self.assertEqual(len(self.pool.complete_event[1]), 1)
        self.assertEqual(self.pool.complete_event[1][0], event)

    def test_add_event_should_store_instant_event_when_phase_is_instant(self):
        """Store instant event when phase is instant."""
        event = mock.Mock(ph=EventConstant.INSTANT_EVENT, tid=1)

        self.pool.add_event(event)

        self.assertEqual(len(self.pool.instance_event[1]), 1)
        self.assertEqual(self.pool.instance_event[1][0], event)

    def test_add_event_should_store_counter_event_when_phase_is_counter(self):
        """Store counter event when phase is counter."""
        event = mock.Mock(ph=EventConstant.COUNTER_EVENT, tid=1)

        self.pool.add_event(event)

        self.assertEqual(len(self.pool.counter_event[1]), 1)
        self.assertEqual(self.pool.counter_event[1][0], event)

    def test_add_event_should_store_meta_event_when_phase_is_meta(self):
        """Store meta event when phase is meta."""
        event = mock.Mock(ph=EventConstant.META_EVENT)

        self.pool.add_event(event)

        self.assertEqual(len(self.pool.meta_event), 1)
        self.assertEqual(self.pool.meta_event[0], event)

    def test_handle_meta_event_should_update_process_name_when_event_is_process_name(self):
        """Update process name when meta event is process name."""
        event = mock.Mock(spec=MetaEvent)
        event.name = EventConstant.PROCESS_NAME
        event.args = {"name": "test_process"}

        self.pool._handle_meta_event(event)

        self.assertEqual(self.pool.name, "test_process")

    def test_handle_meta_event_should_update_thread_mappings_when_event_is_thread_name(self):
        """Update thread mappings when meta event is thread name."""
        event = mock.Mock(spec=MetaEvent)
        event.name = EventConstant.THREAD_NAME
        event.tid = 1
        event.args = {"name": "test_thread"}

        self.pool._handle_meta_event(event)

        self.assertEqual(self.pool.tid_to_name[1], "test_thread")
        self.assertEqual(self.pool.name_to_tid["test_thread"], 1)

    def test_handle_meta_event_should_not_update_thread_mappings_when_tid_is_none(self):
        """Not update thread mappings when tid is None."""
        event = mock.Mock(spec=MetaEvent)
        event.name = EventConstant.THREAD_NAME
        event.tid = None
        event.args = {"name": "test_thread"}

        self.pool._handle_meta_event(event)

        self.assertEqual(len(self.pool.tid_to_name), 0)
        self.assertEqual(len(self.pool.name_to_tid), 0)

    def test_handle_meta_event_should_not_update_thread_mappings_when_name_is_empty(self):
        """Not update thread mappings when thread name is empty."""
        event = mock.Mock(spec=MetaEvent)
        event.name = EventConstant.THREAD_NAME
        event.tid = 1
        event.args = {"name": ""}

        self.pool._handle_meta_event(event)

        self.assertEqual(len(self.pool.tid_to_name), 0)
        self.assertEqual(len(self.pool.name_to_tid), 0)

    def test_add_start_event_should_create_flow_key_when_key_not_exists(self):
        """Create flow key entry when key doesn't exist."""
        event = mock.Mock(spec=BaseEvent)
        flow_key = "test_flow"

        self.pool.add_start_event(flow_key, event)

        self.assertIn(flow_key, self.pool.start_to_end_events_pairs)
        self.assertEqual(len(self.pool.start_to_end_events_pairs[flow_key]["start"]), 1)

    def test_add_end_event_should_create_flow_key_when_key_not_exists(self):
        """Create flow key entry when key doesn't exist."""
        event = mock.Mock(spec=BaseEvent)
        flow_key = "test_flow"

        self.pool.add_end_event(flow_key, event)

        self.assertIn(flow_key, self.pool.start_to_end_events_pairs)
        self.assertEqual(len(self.pool.start_to_end_events_pairs[flow_key]["end"]), 1)

    def test_get_complete_events_should_return_all_events_when_events_exist(self):
        """Return all complete events when events exist."""
        events = [
            mock.Mock(ph=EventConstant.COMPLETE_EVENT, tid=1),
            mock.Mock(ph=EventConstant.COMPLETE_EVENT, tid=2),
            mock.Mock(ph=EventConstant.COMPLETE_EVENT, tid=1)
        ]
        for event in events:
            self.pool.add_event(event)

        complete_events = self.pool.get_complete_events()

        self.assertEqual(len(complete_events), 3)
        self.assertCountEqual(complete_events, events)

    def test_get_instant_events_should_return_all_events_when_events_exist(self):
        """Return all instant events when events exist."""
        events = [
            mock.Mock(ph=EventConstant.INSTANT_EVENT, tid=1),
            mock.Mock(ph=EventConstant.INSTANT_EVENT, tid=2)
        ]
        for event in events:
            self.pool.add_event(event)

        instant_events = self.pool.get_instant_events()

        self.assertEqual(len(instant_events), 2)
        self.assertCountEqual(instant_events, events)

    def test_get_counter_events_should_return_all_events_when_events_exist(self):
        """Return all counter events when events exist."""
        events = [
            mock.Mock(ph=EventConstant.COUNTER_EVENT, tid=1),
            mock.Mock(ph=EventConstant.COUNTER_EVENT, tid=2)
        ]
        for event in events:
            self.pool.add_event(event)

        counter_events = self.pool.get_counter_events()

        self.assertEqual(len(counter_events), 2)
        self.assertCountEqual(counter_events, events)

    def test_get_all_events_should_return_events_in_order_when_events_exist(self):
        """Return all events in correct order when events exist."""
        meta_event = mock.Mock(ph=EventConstant.META_EVENT)
        complete_event = mock.Mock(ph=EventConstant.COMPLETE_EVENT, tid=1)
        instant_event = mock.Mock(ph=EventConstant.INSTANT_EVENT, tid=1)
        counter_event = mock.Mock(ph=EventConstant.COUNTER_EVENT, tid=1)

        events = [complete_event, instant_event, counter_event, meta_event]
        for event in events:
            self.pool.add_event(event)

        all_events = self.pool.get_all_events()

        self.assertEqual(len(all_events), 4)
        self.assertEqual(all_events[0], meta_event)
        self.assertIn(complete_event, all_events[1:])
        self.assertIn(instant_event, all_events[1:])
        self.assertIn(counter_event, all_events[1:])

    def test_get_events_by_tid_should_return_thread_events_when_tid_exists(self):
        """Return all events for specific thread ID when thread exists."""
        tid = 1
        events_tid_1 = [
            mock.Mock(ph=EventConstant.COMPLETE_EVENT, tid=tid),
            mock.Mock(ph=EventConstant.INSTANT_EVENT, tid=tid),
            mock.Mock(ph=EventConstant.COUNTER_EVENT, tid=tid)
        ]
        events_tid_2 = [
            mock.Mock(ph=EventConstant.COMPLETE_EVENT, tid=2),
            mock.Mock(ph=EventConstant.INSTANT_EVENT, tid=2)
        ]

        for event in events_tid_1 + events_tid_2:
            self.pool.add_event(event)

        thread_events = self.pool.get_events_by_tid(tid)

        self.assertEqual(len(thread_events), 3)

    def test_get_events_by_name_should_return_thread_events_when_name_exists(self):
        """Return all events for specific thread name when thread exists."""
        tid = 1
        thread_name = "test_thread"
        self.pool.tid_to_name[tid] = thread_name
        self.pool.name_to_tid[thread_name] = tid

        events = [
            mock.Mock(ph=EventConstant.COMPLETE_EVENT, tid=tid),
            mock.Mock(ph=EventConstant.INSTANT_EVENT, tid=tid)
        ]
        for event in events:
            self.pool.add_event(event)

        thread_events = self.pool.get_events_by_name(thread_name)

        self.assertEqual(len(thread_events), 2)

    def test_get_all_events_with_trace_format_should_return_formatted_events_when_events_exist(self):
        """Return events in Chrome trace format when events exist."""
        events = [
            mock.Mock(spec=BaseEvent),
            mock.Mock(spec=BaseEvent)
        ]
        events[0].tid = 1
        events[1].tid = 2
        events[0].ph = EventConstant.COMPLETE_EVENT
        events[1].ph = EventConstant.INSTANT_EVENT
        events[0].to_trace_format.return_value = {"name": "event1", "ph": EventConstant.COMPLETE_EVENT}
        events[1].to_trace_format.return_value = {"name": "event2", "ph": EventConstant.INSTANT_EVENT}

        for event in events:
            self.pool.add_event(event)

        trace_events = self.pool.get_all_events_with_trace_format()

        self.assertEqual(len(trace_events), 2)
        self.assertEqual(trace_events[0]["name"], "event1")
        self.assertEqual(trace_events[1]["name"], "event2")

    def test_get_events_by_tid_should_return_empty_list_when_tid_not_exists(self):
        """Return empty list when thread ID doesn't exist."""
        events = self.pool.get_events_by_tid(999)
        self.assertEqual(events, [])

    def test_get_events_by_name_should_return_empty_list_when_name_not_exists(self):
        """Return empty list when thread name doesn't exist."""
        events = self.pool.get_events_by_name("unknown_thread")
        self.assertEqual(events, [])

    def test_get_all_tids_should_return_empty_set_when_no_events_exist(self):
        """Return empty set when no events exist."""
        tids = self.pool.get_all_tids()
        self.assertEqual(tids, set())

    def test_get_all_tids_should_return_all_tids_when_events_exist(self):
        """Return set of all thread IDs when events exist."""
        events = [
            mock.Mock(ph=EventConstant.COMPLETE_EVENT, tid=1),
            mock.Mock(ph=EventConstant.INSTANT_EVENT, tid=2),
            mock.Mock(ph=EventConstant.COUNTER_EVENT, tid=3)
        ]
        for event in events:
            self.pool.add_event(event)

        tids = self.pool.get_all_tids()
        self.assertEqual(tids, {1, 2, 3})

    def test_get_start_to_end_flow_pairs_should_return_all_pairs_when_pairs_exist(self):
        """Return all start/end event pairs when pairs exist."""
        event = mock.Mock(spec=BaseEvent)
        flow_key = "test_flow"
        self.pool.add_start_event(flow_key, event)
        self.pool.add_end_event(flow_key, event)

        pairs = self.pool.get_start_to_end_flow_pairs()

        self.assertIn(flow_key, pairs)
        self.assertEqual(len(pairs[flow_key]["start"]), 1)
        self.assertEqual(len(pairs[flow_key]["end"]), 1)

    def test_get_start_to_end_flow_pairs_should_return_empty_dict_when_no_pairs_exist(self):
        """Return empty dictionary when no pairs exist."""
        pairs = self.pool.get_start_to_end_flow_pairs()
        self.assertEqual(pairs, {})


if __name__ == '__main__':
    unittest.main()
