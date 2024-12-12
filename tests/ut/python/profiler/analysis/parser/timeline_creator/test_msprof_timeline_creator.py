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
from mindspore.profiler.analysis.parser.timeline_creator.msprof_timeline_creator import MsprofTimelineCreator


class TestMsprofTimelineCreator(unittest.TestCase):
    """Test cases for MsprofTimelineCreator."""

    def setUp(self):
        """Set up test environment."""
        self.creator = MsprofTimelineCreator()
        # Flow start event
        self.flow_start_event = {
            "name": "flow_start",
            "ph": EventConstant.START_FLOW,
            "cat": EventConstant.HOST_TO_DEVICE_FLOW_CAT,
            "pid": 1,
            "tid": 100,
            "id": "flow1",
            "ts": 1000
        }
        # Flow end event
        self.flow_end_event = {
            "name": "flow_end",
            "ph": EventConstant.END_FLOW,
            "cat": EventConstant.HOST_TO_DEVICE_FLOW_CAT,
            "pid": 2,
            "tid": 200,
            "id": "flow1",
            "ts": 2000,
            "unique_id": "event1"
        }
        # Complete event
        self.complete_event = {
            "name": "MatMul",
            "ph": EventConstant.COMPLETE_EVENT,
            "pid": 2,
            "tid": 200,
            "ts": 2000,
            "dur": 500,
            "unique_id": "event1"
        }
        # Instant event
        self.instant_event = {
            "name": "InstantOp",
            "ph": EventConstant.INSTANT_EVENT,
            "pid": 2,
            "tid": 200,
            "ts": 2500
        }
        # Meta event
        self.meta_event = {
            "name": "process_name",
            "ph": EventConstant.META_EVENT,
            "pid": 2,
            "args": {"name": "test_process"}
        }

    def test_create_should_create_pools_when_input_valid_events(self):
        """Create event pools when input events are valid."""
        events = [self.complete_event, self.instant_event, self.meta_event]
        self.creator.create(events)

        self.assertEqual(len(self.creator.event_pools), 1)
        self.assertIn(2, self.creator.event_pools)

    def test_create_should_not_create_pools_when_input_empty_events(self):
        """Not create event pools when input events are empty."""
        self.creator.create([])

        self.assertEqual(len(self.creator.event_pools), 0)

    # pylint: disable=protected-access
    def test_create_base_events_should_create_events_by_type(self):
        """Create different types of events correctly."""
        events = [self.complete_event, self.instant_event, self.meta_event]
        _, complete_event_map = self.creator._create_base_events(events)

        # Verify complete event map
        self.assertEqual(len(complete_event_map), 1)
        self.assertIn("2-200-2000", complete_event_map)

        # Verify events in pool
        pool = self.creator.event_pools[2]
        self.assertEqual(len(pool.get_complete_events()), 1)
        self.assertEqual(len(pool.get_instant_events()), 1)

    def test_create_base_events_should_create_flow_dict_when_flow_events_exist(self):
        """Create flow dictionary when flow events exist."""
        events = [self.flow_start_event, self.flow_end_event]
        flow_dict, _ = self.creator._create_base_events(events)

        self.assertEqual(len(flow_dict), 1)
        self.assertIn("flow1", flow_dict)
        self.assertIn("start", flow_dict["flow1"])
        self.assertIn("end", flow_dict["flow1"])

    def test_create_acl_to_npu_flow_dict_should_create_flow_mapping(self):
        """Create ACL to NPU flow mapping correctly."""
        events = [self.flow_start_event, self.flow_end_event, self.complete_event]
        flow_dict, complete_event_map = self.creator._create_base_events(events)

        self.creator._create_acl_to_npu_flow_dict(flow_dict, complete_event_map)

        flow_mapping = self.creator.acl_to_npu_flow_dict
        self.assertIn(100, flow_mapping)
        self.assertIn("1000", flow_mapping[100])
        self.assertEqual(len(flow_mapping[100]["1000"]), 1)
        self.assertEqual(flow_mapping[100]["1000"][0].name, "MatMul")

    def test_get_event_pools_should_return_all_pools(self):
        """Return all event pools."""
        events = [self.complete_event, self.meta_event]
        self.creator.create(events)

        pools = self.creator.get_event_pools()

        self.assertEqual(len(pools), 1)
        self.assertIn(2, pools)

    def test_create_acl_to_npu_flow_dict_should_log_warning_when_hardware_event_missing(self):
        """Test warning log when hardware event is missing."""
        # Mock logger to capture warning
        with mock.patch("mindspore.log.warning") as mock_warning:
            # Create flow events with missing hardware event
            self.flow_end_event["unique_id"] = "non_existent_event"
            events = [self.flow_start_event, self.flow_end_event]
            flow_dict, complete_event_map = self.creator._create_base_events(events)

            self.creator._create_acl_to_npu_flow_dict(flow_dict, complete_event_map)
            self.assertEqual(len(self.creator.acl_to_npu_flow_dict), 0)

            mock_warning.assert_called_once_with(
                "Failed to find hardware event for flow end event. Flow ID: flow1, "
                "Unique ID: 2-200-2000"
            )


if __name__ == '__main__':
    unittest.main()
