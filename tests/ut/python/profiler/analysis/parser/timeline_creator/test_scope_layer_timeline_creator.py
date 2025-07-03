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
from decimal import Decimal

from mindspore.profiler.common.constant import EventConstant
from mindspore.profiler.analysis.parser.timeline_event.cpu_op_event import CpuOpCompleteEvent
from mindspore.profiler.analysis.parser.timeline_event.msprof_event import MsprofCompleteEvent
from mindspore.profiler.analysis.parser.timeline_event.fwk_event import FwkCompleteEvent, OpRangeStructField
from mindspore.profiler.common.constant import FileConstant
from mindspore.profiler.analysis.parser.timeline_creator.scope_layer_timeline_creator import (
    ScopeLayerTimelineCreator,
    is_scope_data
)


class TestScopeLayerTimelineCreator(unittest.TestCase):
    """Test cases for ScopeLayerTimelineCreator."""

    def setUp(self):
        """Set up test environment."""
        self.creator = ScopeLayerTimelineCreator()
        self._setup_test_data()

    def _setup_test_data(self):
        """Set up test data."""
        # Create Fwk Event test data
        self.fwk_op_data = self._create_fwk_op_data(
            "Default/network-TrainOneStepCell/network-WithLossCell/MatMul",
            1000000, 2000000, 123
        )

        # Create CpuOpEvent test data
        self.cpu_op_data = {
            "name": "Default/network-TrainOneStepCell/Add",
            "pid": EventConstant.CPU_OP_PID,
            "tid": 1,
            "ts": 2000000,
            "dur": 1
        }

        # Create MsprofEvent test data
        self.msprof_data = {
            "name": "Default/network-TrainOneStepCell/Mul",
            "pid": 1,
            "tid": 1,
            "ts": 3000,
            "dur": 1000
        }

    @staticmethod
    def _create_fwk_op_data(name, start_ns, end_ns, flow_id):
        """Helper method to create MindSporeOpEvent test data."""
        fix_data = (2, flow_id, 0, start_ns, end_ns, 1, 0, 0, 0, 1, False, False, True)

        return {
            FileConstant.FIX_SIZE_DATA: fix_data,
            OpRangeStructField.NAME.value: name,  # name
            OpRangeStructField.FULL_NAME.value: name,  # full_name
            OpRangeStructField.MODULE_GRAPH.value: "",  # module_graph
            OpRangeStructField.EVENT_GRAPH.value: "",  # event_graph
            OpRangeStructField.CUSTOM_INFO.value: "type:MatMul"  # custom_info
        }

    def test_create_should_create_scope_layers_when_processing_mindspore_op(self):
        """Should create correct scope layers when processing MindSpore operation."""
        with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                        side_effect=lambda x: Decimal(x) / Decimal('1000')):
            op = FwkCompleteEvent(self.fwk_op_data)
            self.creator.create([op])

            pool = self.creator.event_pools.get(int(EventConstant.SCOPE_LAYER_PID))
            self.assertIsNotNone(pool)
            events = pool.get_complete_events()

            # Should create scope layers: Default, network-TrainOneStepCell, network-WithLossCell
            self.assertEqual(len(events), 3)
            self.assertEqual(events[0].name, "Default")
            self.assertEqual(events[1].name, "network-TrainOneStepCell")
            self.assertEqual(events[2].name, "network-WithLossCell")

            # Verify timestamps
            self.assertEqual(events[0].ts, Decimal('1000'))
            self.assertEqual(events[0].dur, Decimal('1000'))
            self.assertEqual(events[1].ts, Decimal('1000'))
            self.assertEqual(events[1].dur, Decimal('1000'))
            self.assertEqual(events[2].ts, Decimal('1000'))
            self.assertEqual(events[2].dur, Decimal('1000'))

    def test_create_should_create_scope_layers_when_processing_cpu_op(self):
        """Should create correct scope layers when processing CPU operation."""
        op = CpuOpCompleteEvent(self.cpu_op_data)
        self.creator.create([op])

        pool = self.creator.event_pools.get(int(EventConstant.SCOPE_LAYER_PID))
        self.assertIsNotNone(pool)
        events = pool.get_complete_events()

        # Should create scope layers: Default, network-TrainOneStepCell
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].name, "Default")
        self.assertEqual(events[1].name, "network-TrainOneStepCell")

        # Verify timestamps
        self.assertEqual(events[0].ts, Decimal('2000'))
        self.assertEqual(events[0].dur, Decimal('1000'))
        self.assertEqual(events[1].ts, Decimal('2000'))
        self.assertEqual(events[1].dur, Decimal('1000'))

    def test_create_should_create_scope_layers_when_processing_msprof_event(self):
        """Should create correct scope layers when processing Msprof event."""
        op = MsprofCompleteEvent(self.msprof_data)
        self.creator.create([op])

        pool = self.creator.event_pools.get(int(EventConstant.SCOPE_LAYER_PID))
        self.assertIsNotNone(pool)
        events = pool.get_complete_events()

        # Should create scope layers: Default, network-TrainOneStepCell
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].name, "Default")
        self.assertEqual(events[1].name, "network-TrainOneStepCell")

        # Verify timestamps
        self.assertEqual(events[0].ts, Decimal('3000'))
        self.assertEqual(events[0].dur, Decimal('1000'))
        self.assertEqual(events[1].ts, Decimal('3000'))
        self.assertEqual(events[1].dur, Decimal('1000'))

    def test_create_should_skip_event_when_start_time_after_previous_end(self):
        """Should skip event when its start time is after previous event's end time."""
        with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                        side_effect=lambda x: Decimal(x) / Decimal('1000')):
            op1 = FwkCompleteEvent(self._create_fwk_op_data(
                "Default/network-TrainOneStepCell/MatMul",
                1000000, 2000000, 123
            ))
            # Create an event with start time before op1's end time
            op2 = FwkCompleteEvent(self._create_fwk_op_data(
                "Default/network-TrainOneStepCell/Add",
                1500000, 3000000, 124
            ))

            self.creator.create([op1, op2])

            pool = self.creator.event_pools.get(int(EventConstant.SCOPE_LAYER_PID))
            events = pool.get_complete_events()

            # Verify op2 is skipped, only op1's layers exist
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].name, "Default")
            self.assertEqual(events[1].name, "network-TrainOneStepCell")
            self.assertEqual(events[0].ts, Decimal('1000'))
            self.assertEqual(events[0].dur, Decimal('1000'))
            self.assertEqual(events[1].ts, Decimal('1000'))
            self.assertEqual(events[1].dur, Decimal('1000'))

    def test_create_should_merge_layers_when_scope_names_match(self):
        """Should merge layers when scope names match and events are sequential."""
        with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                        side_effect=lambda x: Decimal(x) / Decimal('1000')):
            # Two events with same scope hierarchy
            op1 = FwkCompleteEvent(self._create_fwk_op_data(
                "Default/network-TrainOneStepCell/network-WithLossCell/MatMul",
                1000000, 2000000, 123
            ))
            op2 = FwkCompleteEvent(self._create_fwk_op_data(
                "Default/network-TrainOneStepCell/network-WithLossCell/Add",
                2500000, 3500000, 124
            ))

            self.creator.create([op1, op2])

            pool = self.creator.event_pools.get(int(EventConstant.SCOPE_LAYER_PID))
            events = pool.get_complete_events()

            # Verify layers are merged
            self.assertEqual(len(events), 3)  # Default, TrainOneStepCell, WithLossCell
            self.assertEqual(events[0].name, "Default")
            self.assertEqual(events[1].name, "network-TrainOneStepCell")
            self.assertEqual(events[2].name, "network-WithLossCell")

            # Verify merged timestamps
            self.assertEqual(events[0].ts, Decimal('1000'))
            self.assertEqual(events[0].dur, Decimal('2500'))
            self.assertEqual(events[1].ts, Decimal('1000'))
            self.assertEqual(events[1].dur, Decimal('2500'))
            self.assertEqual(events[2].ts, Decimal('1000'))
            self.assertEqual(events[2].dur, Decimal('2500'))

    def test_create_should_create_separate_layers_when_scope_names_differ(self):
        """Should create separate layers when scope names differ."""
        with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                        side_effect=lambda x: Decimal(x) / Decimal('1000')):
            op1 = FwkCompleteEvent(self._create_fwk_op_data(
                "Default/network-TrainOneStepCell/network-WithLossCell/MatMul",
                1000000, 2000000, 123
            ))
            op2 = FwkCompleteEvent(self._create_fwk_op_data(
                "Default/network-TrainOneStepCell/network-Optimizer/Add",
                2500000, 3500000, 124
            ))
            self.creator.create([op1, op2])

            pool = self.creator.event_pools.get(int(EventConstant.SCOPE_LAYER_PID))
            events = pool.get_complete_events()

            # Verify separate layers are created
            self.assertEqual(len(events), 4)  # All layers from both hierarchies
            event_names = {event.name for event in events}
            self.assertIn("Default", event_names)
            self.assertIn("network-TrainOneStepCell", event_names)
            self.assertIn("network-WithLossCell", event_names)
            self.assertIn("network-Optimizer", event_names)

    def test_create_should_return_empty_when_input_is_empty(self):
        """Should return empty result when input list is empty."""
        self.creator.create([])
        pool = self.creator.event_pools.get(int(EventConstant.SCOPE_LAYER_PID))
        self.assertIsNone(pool)

    # pylint: disable=protected-access
    def test_parse_scope_data_should_return_correct_data_when_input_event_with_parent(self):
        """Should return correct scope data when event has parent event."""
        with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                        side_effect=lambda x: Decimal(x) / Decimal('1000')):
            main_event = MsprofCompleteEvent(self.msprof_data)
            parent_event = FwkCompleteEvent(self._create_fwk_op_data(
                "Default/network-TrainOneStepCell/network-WithLossCell/Op2",
                1000000, 2000000, 123
            ))
            main_event.parent = parent_event

            result = self.creator._parse_scope_data(main_event)

            self.assertIsNotNone(result)
            scope_names, start_time, dur_time = result
            self.assertEqual(
                scope_names,
                ["Default", "network-TrainOneStepCell", "network-WithLossCell"]
            )
            self.assertEqual(start_time, Decimal('3000'))
            self.assertEqual(dur_time, Decimal('1000'))

    def test_parse_scope_data_should_return_correct_data_when_input_event_without_parent(self):
        """Should return correct scope data when event has no parent."""
        with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                        side_effect=lambda x: Decimal(x) / Decimal('1000')):
            event = FwkCompleteEvent(self._create_fwk_op_data(
                "Default/network-TrainOneStepCell/Op1",
                1000000, 2000000, 123
            ))

            result = self.creator._parse_scope_data(event)

            self.assertIsNotNone(result)
            scope_names, start_time, dur_time = result
            self.assertEqual(scope_names, ["Default", "network-TrainOneStepCell"])
            self.assertEqual(start_time, Decimal('1000'))
            self.assertEqual(dur_time, Decimal('1000'))

    def test_parse_scope_data_should_return_none_when_input_invalid_event(self):
        """Should return None when event has invalid scope data."""
        # Test with invalid scope name
        invalid_event = FwkCompleteEvent(self._create_fwk_op_data(
            "InvalidScope",
            1000000, 2000000, 123
        ))
        result = self.creator._parse_scope_data(invalid_event)
        self.assertIsNone(result)

        # Test with empty scope name
        empty_event = FwkCompleteEvent(self._create_fwk_op_data(
            "", 1000000, 2000000, 123
        ))
        result = self.creator._parse_scope_data(empty_event)
        self.assertIsNone(result)

    def test_is_scope_data_should_return_true_when_valid_scope_event(self):
        """Should return True when event is valid scope event."""
        cpu_event = CpuOpCompleteEvent({"name": "Default/network/MatMul"})
        msprof_event = MsprofCompleteEvent({"name": "Default/network/MatMul"})
        fwk_event = FwkCompleteEvent(self._create_fwk_op_data(
            "Default/network/MatMul",
            1000000, 2000000, 123
        ))
        self.assertTrue(is_scope_data(cpu_event))
        self.assertTrue(is_scope_data(msprof_event))
        self.assertTrue(is_scope_data(fwk_event))

    def test_is_scope_data_should_return_false_when_invalid_scope_event(self):
        """Should return False when event is invalid scope event."""
        cpu_event = CpuOpCompleteEvent({"name": "invalid_name"})
        msprof_event = MsprofCompleteEvent({"name": "invalid_name"})
        fwk_event = FwkCompleteEvent(self._create_fwk_op_data(
            "invalid_name",
            1000000, 2000000, 123
        ))
        self.assertFalse(is_scope_data(cpu_event))
        self.assertFalse(is_scope_data(msprof_event))
        self.assertFalse(is_scope_data(fwk_event))

    def test_get_chrome_trace_data_should_return_events_when_scope_layers_exist(self):
        """Should return chrome trace events when scope layers exist."""
        with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                        side_effect=lambda x: Decimal(x) / Decimal('1000')):
            op = FwkCompleteEvent(self._create_fwk_op_data(
                "Default/network-TrainOneStepCell/network-WithLossCell/MatMul",
                1000000, 2000000, 123
            ))
            self.creator.create([op])

            events = self.creator.get_chrome_trace_data()

            # Verify events contain both metadata and scope events
            # 3 scope layer events and 6 thread metadata events and 2 process metadata events
            self.assertEqual(len(events), 11)

            # Verify scope event names
            event_names = {event.get('name') for event in events if 'name' in event}
            self.assertIn('Default', event_names)
            self.assertIn('network-TrainOneStepCell', event_names)
            self.assertIn('network-WithLossCell', event_names)

            # Verify event format
            scope_event = next(event for event in events if event.get('name') == 'Default')
            self.assertEqual(scope_event['ph'], EventConstant.COMPLETE_EVENT)
            self.assertEqual(scope_event['pid'], int(EventConstant.SCOPE_LAYER_PID))

    def test_get_chrome_trace_data_should_return_empty_when_no_scope_layers(self):
        """Should return empty list when no scope layers exist."""
        events = self.creator.get_chrome_trace_data()
        self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main()
