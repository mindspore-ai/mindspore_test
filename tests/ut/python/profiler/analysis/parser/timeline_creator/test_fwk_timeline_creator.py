import unittest
from unittest import mock
from decimal import Decimal

from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.common.constant import EventConstant, FileConstant, TimelineLayerName
from mindspore.profiler.analysis.parser.timeline_creator.fwk_timeline_creator import FwkTimelineCreator
from mindspore.profiler.analysis.parser.timeline_event.fwk_event import OpRangeStructField


class TestFwkTimelineCreator(unittest.TestCase):
    """Test cases for FwkTimelineCreator."""

    def setUp(self):
        """Set up test environment."""
        self.creator = FwkTimelineCreator()
        self.normal_data = {
            FileConstant.FIX_SIZE_DATA: [2, 0, 1, 1000, 2000, 1, 2, 2, 3, 1, False, False, True],
            OpRangeStructField.NAME.value: "MatMul",
            OpRangeStructField.FULL_NAME.value: "MatMul",
            OpRangeStructField.MODULE_GRAPH.value: "",
            OpRangeStructField.EVENT_GRAPH.value: "",
            OpRangeStructField.CUSTOM_INFO.value: "key1:value1"
        }
        self.instant_data = {
            FileConstant.FIX_SIZE_DATA: [2, 0, 1, 1000, 1000, 1, 2, 2, 3, 1, False, False, True],
            OpRangeStructField.NAME.value: "InstantOp",
            OpRangeStructField.FULL_NAME.value: "InstantOp"
        }
        self.flow_data = {
            FileConstant.FIX_SIZE_DATA: [2, 123, 1, 1000, 2000, 1, 2, 2, 3, 1, False, False, True],
            OpRangeStructField.NAME.value: EventConstant.FLOW_OP,
            OpRangeStructField.FULL_NAME.value: EventConstant.FLOW_OP
        }
        self.flow_end_data = {
            FileConstant.FIX_SIZE_DATA: [2, 123, 1, 2000, 3000, 1, 2, 2, 3, 1, False, False, True],
            OpRangeStructField.NAME.value: "FlowEnd",
            OpRangeStructField.FULL_NAME.value: "FlowEnd"
        }

    def test_create_should_create_pool_when_input_valid_data(self):
        """Create event pool when input data is valid."""
        self.creator.create([self.normal_data])

        self.assertIn(EventConstant.MINDSPORE_PID, self.creator.event_pools)
        pool = self.creator.event_pools[EventConstant.MINDSPORE_PID]
        events = pool.get_all_events()

        # 1 complete event + 3 process meta events + 2 thread meta events
        self.assertEqual(len(events), 6)

    def test_create_should_not_create_pool_when_input_empty_data(self):
        """Not create event pool when input data is empty."""
        self.creator.create([])
        self.assertEqual(len(self.creator.event_pools), 0)

    # pylint: disable=protected-access
    def test_create_base_events_should_create_events_with_different_types(self):
        """Create different types of events based on input data."""
        pool = self.creator.event_pools.setdefault(
            EventConstant.MINDSPORE_PID,
            TimelineEventPool(EventConstant.MINDSPORE_PID)
        )

        # Test complete event
        self.creator._create_base_events(pool, [self.normal_data])
        complete_events = pool.get_complete_events()
        self.assertEqual(len(complete_events), 1)
        self.assertEqual(complete_events[0].name, "MatMul")

        # Test instant event
        self.creator._create_base_events(pool, [self.instant_data])
        instant_events = pool.get_instant_events()
        self.assertEqual(len(instant_events), 1)
        self.assertEqual(instant_events[0].name, "InstantOp")

        # Test flow events
        self.creator._create_base_events(pool, [self.flow_data, self.flow_end_data])
        flow_pairs = pool.get_start_to_end_flow_pairs()
        self.assertIn("123", flow_pairs)
        self.assertEqual(len(flow_pairs["123"]["start"]), 1)
        self.assertEqual(len(flow_pairs["123"]["end"]), 1)

    def test_create_base_events_should_filter_event_when_start_time_is_zero(self):
        """Filter event when start time is zero."""
        abnormal_data = dict(self.normal_data)
        abnormal_data[FileConstant.FIX_SIZE_DATA][OpRangeStructField.START_TIME_NS.value] = 0

        pool = self.creator.event_pools.setdefault(
            EventConstant.MINDSPORE_PID,
            TimelineEventPool(EventConstant.MINDSPORE_PID)
        )
        self.creator._create_base_events(pool, [abnormal_data])

        complete_events = pool.get_complete_events()
        self.assertEqual(len(complete_events), 0)

    def test_create_meta_events_should_create_process_and_thread_meta_events(self):
        """Create both process and thread meta events correctly."""
        pool = self.creator.event_pools.setdefault(
            EventConstant.MINDSPORE_PID,
            TimelineEventPool(EventConstant.MINDSPORE_PID)
        )
        # Add some events to create threads
        self.creator._create_base_events(pool, [self.normal_data, self.instant_data])
        self.creator._create_meta_event(pool)

        all_events = pool.get_all_events()

        # Check process meta events
        self.assertTrue(any(
            event.name == EventConstant.PROCESS_NAME and
            event.args["name"] == TimelineLayerName.MINDSPORE.value
            for event in all_events
        ))
        self.assertTrue(any(
            event.name == EventConstant.PROCESS_SORT and
            event.args["sort_index"] == EventConstant.MINDSPORE_SORT_IDX
            for event in all_events
        ))
        self.assertTrue(any(
            event.name == EventConstant.PROCESS_LABEL and
            event.args["labels"] == EventConstant.CPU_LABEL
            for event in all_events
        ))

        # Check thread meta events
        thread_names = {
            event.args["name"] for event in all_events
            if event.name == EventConstant.THREAD_NAME
        }
        self.assertEqual(thread_names, {"Thread 2"})

    def test_get_chrome_trace_data_should_return_formatted_events(self):
        """Return formatted events in Chrome trace format."""
        with mock.patch(
                "mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                side_effect=lambda x: Decimal(x) / Decimal('1000')
        ):
            self.creator.create([self.normal_data])
            trace_events = self.creator.get_chrome_trace_data()
            self.assertTrue(any(event["name"] == "MatMul" for event in trace_events))
            self.assertTrue(any(event["name"] == EventConstant.PROCESS_NAME for event in trace_events))

    def test_get_chrome_trace_data_should_return_empty_when_no_pools(self):
        """Return empty list when no event pools exist."""
        trace_events = self.creator.get_chrome_trace_data()
        self.assertEqual(trace_events, [])

    def test_get_event_pools_should_return_all_pools(self):
        """Return all event pools."""
        self.creator.create([self.normal_data])
        pools = self.creator.get_event_pools()
        self.assertEqual(len(pools), 1)
        self.assertIn(EventConstant.MINDSPORE_PID, pools)


if __name__ == '__main__':
    unittest.main()
