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
from unittest.mock import patch
from decimal import Decimal

from mindspore.profiler.common.log import ProfilerLogger
from mindspore.profiler.common.constant import EventConstant, TimelineLayerName, FileConstant
from mindspore.profiler.analysis.parser.timeline_event.fwk_event import FwkCompleteEvent
from mindspore.profiler.analysis.parser.timeline_event.msprof_event import MsprofCompleteEvent
from mindspore.profiler.analysis.parser.timeline_event.timeline_event_pool import TimelineEventPool
from mindspore.profiler.analysis.parser.timeline_assembly_factory.ascend_timeline_assembler import (
    AscendTimelineAssembler
)
from mindspore.profiler.analysis.parser.timeline_event.fwk_event import OpRangeStructField


# pylint: disable=protected-access
class TestAscendTimelineAssembler(unittest.TestCase):
    """Test cases for AscendTimelineAssembler.

    This test suite verifies the timeline assembly functionality for three different execution modes:
    1. Kernel by kernel (KBK) mode: Tests the assembly of kernel launch operations
    2. Graph mode: Tests the assembly of graph execution operations
    3. PyNative mode: Tests the assembly of PyNative operations

    Each mode has its own specific timeline events and flow relationships that need to be correctly
    assembled and verified.
    """

    @patch.object(ProfilerLogger, 'init')
    @patch.object(ProfilerLogger, 'get_instance')
    def setUp(self, mock_get_instance, mock_init):
        """Set up test environment."""
        mock_logger = unittest.mock.MagicMock()
        mock_get_instance.return_value = mock_logger
        self.assembler = AscendTimelineAssembler()
        self._setup_test_data()

    def _setup_test_data(self):
        """Set up test data for different execution modes."""
        self.fwk_kbk_op_data = self._create_fwk_kbk_model_data()
        self.fwk_graph_op_data = self._create_fwk_graph_model_data()
        self.fwk_pynative_op_data = self._create_fwk_pynative_model_data()
        self.mstx_op_fwk_data = self._create_mstx_fwk_data()
        self.cpu_op_data = ["Default/network-TrainOneStepCell/Add;Add;1500,200,2"]
        self.msprof_data = self._create_msprof_data()
        self.msprof_tx_data = self._create_msprof_tx_data()

    def _create_base_fwk_data(self, op_name, op_type, start_us, end_us, thread_id):
        """Create base framework operation data."""
        return {
            FileConstant.FIX_SIZE_DATA: (thread_id, 123, 0, start_us, end_us, 1, 1, 2, 3, 1, False, False, True),
            OpRangeStructField.NAME.value: op_name,
            OpRangeStructField.FULL_NAME.value: op_name,
            OpRangeStructField.MODULE_GRAPH.value: "",
            OpRangeStructField.EVENT_GRAPH.value: "",
            OpRangeStructField.CUSTOM_INFO.value: f"type:{op_type}"
        }

    def _create_fwk_kbk_model_data(self):
        """Create framework operation data for KBK mode."""
        return [self._create_base_fwk_data("KernelLaunch::Default/network-TrainOneStepCell/MatMul",
                                           "MatMul", 1000000, 2000000, 100)]

    def _create_fwk_graph_model_data(self):
        """Create framework operation data for Graph mode."""
        return [self._create_base_fwk_data(
            "Ascend::RunGraph::GeRunGraph_kernel_graph()",
            "Graph", 1000000, 2000000, 100)]

    def _create_fwk_pynative_model_data(self):
        """Create framework operation data for PyNative mode."""
        return [
            self._create_base_fwk_data(
                "PynativeFrameWork::LaunchTask::pyboost",
                "MatMul", 1000000, 2000000, 100),
            self._create_base_fwk_data(
                EventConstant.FLOW_OP,
                "flow_op", 500000, 1000000, 2)
        ]

    def _create_mstx_fwk_data(self):
        """Create framework operation data for mstx."""
        return [
            self._create_base_fwk_data(
                "Mstx_mark_op",
                "mstx", 1000000, 2000000, 100),
            self._create_base_fwk_data(
                "Mstx_range_start_op",
                "mstx", 2500000, 3500000, 100)
        ]

    def _create_msprof_data(self):
        """Create Msprof operation data."""
        self.msprof_cann_data = self._create_msprof_event("node@launch", 100, 100, 1500)
        self.msprof_hardware_data = self._create_msprof_event(
            "Default/network-TrainOneStepCell/Mul", 200, 200, 3000)
        self.msprof_cann_meta_data = {
            "name": EventConstant.PROCESS_NAME,
            "ph": EventConstant.META_EVENT,
            "pid": 100,
            "tid": 0,
            "args": {"name": TimelineLayerName.CANN.value}
        }
        self.msprof_meta_data = {
            "name": EventConstant.PROCESS_NAME,
            "ph": EventConstant.META_EVENT,
            "pid": 200,
            "tid": 0,
            "args": {"name": TimelineLayerName.ASCEND_HARDWARE.value}
        }
        self.flow_start_event = self._create_flow_event(
            "flow_start", EventConstant.START_FLOW, 100, 100, "flow1", 2000)
        self.flow_end_event = self._create_flow_event(
            "flow_end", EventConstant.END_FLOW, 200, 200, "flow1", 3000)

        return [self.msprof_cann_data, self.msprof_hardware_data, self.msprof_meta_data, self.msprof_cann_meta_data,
                self.flow_start_event, self.flow_end_event]

    def _create_msprof_tx_data(self):
        """Create Msprof operation data."""
        msprof_tx_mark_event = self._create_msprof_event("mark", 100, 100, 1500)
        msprof_hardware_tx_mark_event = self._create_msprof_event(
            "mark", 200, 200, 2000)

        msprof_tx_mark_flow_start_event = self._create_flow_event(
            "mark_flow_start", EventConstant.START_FLOW, 100, 100, "flow1", 1700)
        msprof_tx_mark_flow_end_event = self._create_flow_event(
            "mark_flow_end", EventConstant.END_FLOW, 200, 200, "flow1", 2000)

        msprof_tx_range_event = self._create_msprof_event("range", 100, 100, 3000)
        msprof_hardware_tx_range_event = self._create_msprof_event(
            "range", 200, 200, 3500)

        msprof_tx_range_flow_start_event = self._create_flow_event(
            "range_flow_start", EventConstant.START_FLOW, 100, 100, "flow2", 3200)
        msprof_tx_range_flow_end_event = self._create_flow_event(
            "range_flow_end", EventConstant.END_FLOW, 200, 200, "flow2", 3500)

        msprof_tx_meta_data = {
            "name": EventConstant.PROCESS_NAME,
            "ph": EventConstant.META_EVENT,
            "pid": 100,
            "tid": 0,
            "args": {"name": 'python'}
        }

        msprof_meta_data = {
            "name": EventConstant.PROCESS_NAME,
            "ph": EventConstant.META_EVENT,
            "pid": 200,
            "tid": 0,
            "args": {"name": TimelineLayerName.ASCEND_HARDWARE.value}
        }

        return [msprof_tx_mark_event, msprof_hardware_tx_mark_event, msprof_tx_mark_flow_start_event,
                msprof_tx_mark_flow_end_event, msprof_tx_range_event, msprof_hardware_tx_range_event,
                msprof_tx_range_flow_start_event, msprof_tx_range_flow_end_event, msprof_tx_meta_data,
                msprof_meta_data]

    def _create_msprof_event(self, name, pid, tid, ts, dur=1000, ph=EventConstant.COMPLETE_EVENT):
        """Create Msprof event data."""
        return {
            "name": name,
            "ph": ph,
            "pid": pid,
            "tid": tid,
            "ts": ts,
            "dur": dur,
        }

    def _create_flow_event(self, name, ph, pid, tid, flow_id, ts):
        """Create flow event data."""
        return {
            "name": name,
            "ph": ph,
            "cat": EventConstant.HOST_TO_DEVICE_FLOW_CAT,
            "pid": pid,
            "tid": tid,
            "id": flow_id,
            "ts": ts
        }

    def test_init_creators_should_initialize_all_creators_when_created(self):
        """Should initialize all creators when assembler is created."""
        self.assertIsNotNone(self.assembler._fwk_creator)
        self.assertIsNotNone(self.assembler._cpu_op_creator)
        self.assertIsNotNone(self.assembler._msprof_creator)
        self.assertIsNotNone(self.assembler._scope_layer_creator)

    def test_assemble_events_should_create_events_when_input_valid_data_in_kbk_model(self):
        """Should create events for all operation types when input data is valid in kbk model."""
        with (mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                         side_effect=lambda x: Decimal(x) / Decimal('1000'))):
            data = {
                "mindspore_op_list": self.fwk_kbk_op_data,
                "cpu_op_lines": self.cpu_op_data,
                "msprof_timeline": self.msprof_data
            }

            self.assembler.assemble(data)

            container = self.assembler.trace_view_container

            # Verify MindSpore events
            mindspore_pool = container.get_pool_by_name(TimelineLayerName.MINDSPORE.value)
            self.assertIsNotNone(mindspore_pool)
            mindspore_events = mindspore_pool.get_complete_events()
            self.assertEqual(len(mindspore_events), 1)
            self.assertEqual(mindspore_events[0].name, "KernelLaunch::Default/network-TrainOneStepCell/MatMul")

            # Verify CPU events
            cpu_pool = container.get_pool_by_name(TimelineLayerName.CPU_OP.value)
            self.assertIsNotNone(cpu_pool)
            cpu_events = cpu_pool.get_complete_events()
            self.assertEqual(len(cpu_events), 1)
            self.assertEqual(cpu_events[0].name, "Default/network-TrainOneStepCell/Add")

            # Verify Hardware events
            hardware_pool = container.get_pool_by_name(TimelineLayerName.ASCEND_HARDWARE.value)
            self.assertIsNotNone(hardware_pool)
            hardware_events = hardware_pool.get_complete_events()
            self.assertEqual(len(hardware_events), 1)
            self.assertEqual(hardware_events[0].name, "Default/network-TrainOneStepCell/Mul")

            # 6 cpu op events and 6 msprof events and 6 fwk events and 8 scope layer events and 2 flow events
            self.assertEqual(len(container.get_trace_view()), 28)

    def test_assemble_events_should_create_events_when_input_valid_data_in_graph_model(self):
        """Should create events for all operation types when input data is valid in graph model."""
        with (mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                         side_effect=lambda x: Decimal(x) / Decimal('1000'))):
            data = {
                "mindspore_op_list": self.fwk_graph_op_data,
                "cpu_op_lines": self.cpu_op_data,
                "msprof_timeline": self.msprof_data
            }

            self.assembler.assemble(data)

            container = self.assembler.trace_view_container
            # Verify trace view: 6 cpu op events and 6 msprof events and 6 fwk events and 8 scope layer events
            self.assertEqual(len(container.get_trace_view()), 26)

    def test_assemble_events_should_create_events_when_input_valid_data_in_pynative_model(self):
        """Should create events for all operation types when input data is valid in pynative model."""
        with (mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                         side_effect=lambda x: Decimal(x) / Decimal('1000'))):
            data = {
                "mindspore_op_list": self.fwk_pynative_op_data,
                "cpu_op_lines": self.cpu_op_data,
                "msprof_timeline": self.msprof_data
            }

            self.assembler.assemble(data)

            container = self.assembler.trace_view_container
            # 6 cpu op events and 6 msprof events and 7 fwk events and 7 scope layer events and 4 flow events
            self.assertEqual(len(container.get_trace_view()), 30)

    def test_assemble_events_should_create_events_when_input_valid_data_in_mstx_with_level_none(self):
        """Should create events for all operation types when input data is valid in mstx with level none."""
        with (mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                         side_effect=lambda x: Decimal(x) / Decimal('1000'))):
            data = {
                "mindspore_op_list": self.mstx_op_fwk_data,
                "cpu_op_lines": [],
                "msprof_timeline": self.msprof_tx_data
            }

            self.assembler.assemble(data)

            container = self.assembler.trace_view_container
            # 10 msprof events and 7 fwk events and 4 flow events
            self.assertEqual(len(container.get_trace_view()), 21)

    def test_assemble_should_handle_empty_input_when_data_is_empty(self):
        """Should handle empty input data when input dictionary is empty."""
        empty_data = {
            "mindspore_op_list": [],
            "cpu_op_lines": [],
            "msprof_timeline": []
        }
        self.assembler.assemble(empty_data)

        container = self.assembler.trace_view_container
        self.assertEqual(len(container.get_all_pools()), 0)
        self.assertEqual(len(container.get_trace_view()), 0)

    def test_create_fwk_to_hardware_flow_should_create_flow_events_when_hardware_events_exist(self):
        """Should create flow events between framework and hardware events."""
        hardware_event = MsprofCompleteEvent(self.msprof_hardware_data)
        launch_event = FwkCompleteEvent(self._create_base_fwk_data(
            "KernelLaunch::Default/network-TrainOneStepCell/MatMul",
            "MatMul", 1000000, 2000000, 100))

        acl_to_npu_flow_dict = {
            100: {"1500": [hardware_event]}  # tid as key, timestamp as key
        }
        fwk_launch_op_list = {
            100: [launch_event]  # tid as key
        }
        self.assembler.trace_view_container.kernel_launch_op_event = fwk_launch_op_list

        with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                        side_effect=lambda x: Decimal(x) / Decimal('1000')):
            with mock.patch.object(self.assembler._msprof_creator, 'get_acl_to_npu_flow_dict',
                                   return_value=acl_to_npu_flow_dict):
                flow_dicts = self.assembler._create_fwk_to_hardware_flow()
                self.assertEqual(len(flow_dicts), 2)

                # Verify flow event properties
                start_flow = next(
                    flow_dict for flow_dict in flow_dicts
                    if flow_dict.get("ph") == EventConstant.START_FLOW
                )
                end_flow = next(
                    flow_dict for flow_dict in flow_dicts
                    if flow_dict.get("ph") == EventConstant.END_FLOW
                )

                self.assertEqual(start_flow["cat"], EventConstant.MINDSPORE_NPU_FLOW_CAT)
                self.assertEqual(end_flow["cat"], EventConstant.MINDSPORE_NPU_FLOW_CAT)
                self.assertEqual(start_flow["id"], "3000")  # Using hardware event ts as id
                self.assertEqual(end_flow["id"], "3000")

    def test_create_fwk_to_mstx_flow_should_create_flow_events_when_mstx_events_exist(self):
        """Should create flow events between framework and mstx events.
        Timeline visualization:
            Framework Layer:
                   ts(1000)                                                          te(2000)
            tid1    |-------------------- Mstx_range_event_1 ---------------------------|
                                                           ts(1600)            te(1900)
            tid2                                           |-- Mstx_range_event_2 --|

            MSTX Layer:
                        ts(1200)                                                te(1900)
            tid1        |------------------- Mstx_range_event_1 -------------------|
                                                           ts(1650)        te(1850)
            tid2                                            |-Mstx_range_event_2-|
        """

        fwk_mstx_event_1 = FwkCompleteEvent(self._create_base_fwk_data(
            "Mstx_range_event_1",
            "mstx", 1000000, 2000000, 100))

        mstx_event_1 = MsprofCompleteEvent({
            "name": "Mstx_range_event_1",
            "ph": EventConstant.COMPLETE_EVENT,
            "pid": 100,
            "tid": 100,
            "ts": 1200,
            "dur": 700
        })

        fwk_mstx_event_2 = FwkCompleteEvent(self._create_base_fwk_data(
            "Mstx_range_event_2",
            "mstx", 1600000, 1900000, 200))

        mstx_event_2 = MsprofCompleteEvent({
            "name": "Mstx_range_event_2",
            "ph": EventConstant.COMPLETE_EVENT,
            "pid": 100,
            "tid": 200,
            "ts": 1650,
            "dur": 200
        })

        # Create and setup MSTX pool
        mstx_pool = TimelineEventPool('python')
        mstx_pool.add_event(mstx_event_1)
        mstx_pool.add_event(mstx_event_2)

        # Create and setup framework pool
        fwk_pool = TimelineEventPool(TimelineLayerName.MINDSPORE.value)
        fwk_pool.add_event(fwk_mstx_event_1)
        fwk_pool.add_event(fwk_mstx_event_2)

        with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                        side_effect=lambda x: Decimal(x) / Decimal('1000')):
            flow_dicts = self.assembler._create_fwk_to_mstx_flow(mstx_pool, fwk_pool)

            # Verify flow events were created
            self.assertEqual(len(flow_dicts), 4)

    def test_create_fwk_to_fwk_flow_should_create_flow_events_when_matching_events_exist(self):
        """Should create flow events between framework events."""
        with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                        side_effect=lambda x: Decimal(x) / Decimal('1000')):
            data = {
                "mindspore_op_list": self.fwk_pynative_op_data,
                "msprof_timeline": self.msprof_data
            }

            self.assembler._assemble_basic_events(data)
            pool = self.assembler.trace_view_container.get_pool_by_name(TimelineLayerName.MINDSPORE.value)
            flow_dicts = self.assembler._create_fwk_to_fwk_flow(pool)

            self.assertEqual(len(flow_dicts), 2)

            # Verify flow event properties
            start_flow = next(
                flow_dict for flow_dict in flow_dicts
                if flow_dict.get("ph") == EventConstant.START_FLOW
            )
            end_flow = next(
                flow_dict for flow_dict in flow_dicts
                if flow_dict.get("ph") == EventConstant.END_FLOW
            )
            self.assertEqual(start_flow["cat"], EventConstant.MINDSPORE_SELF_FLOW_CAT)
            self.assertEqual(end_flow["cat"], EventConstant.MINDSPORE_SELF_FLOW_CAT)
            self.assertEqual(start_flow["id"], "123")
            self.assertEqual(end_flow["id"], "123")

    def test_create_flow_events_should_create_all_types_of_flows_when_events_exist(self):
        """Should create all types of flow events when corresponding events exist."""
        with (mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                         side_effect=lambda x: Decimal(x) / Decimal('1000'))):
            data = {
                "mindspore_op_list": self.fwk_pynative_op_data,
                "msprof_timeline": self.msprof_data
            }

            self.assembler._assemble_basic_events(data)
            self.assembler._assemble_flow_events()

            container = self.assembler.trace_view_container
            trace_events = container.get_trace_view()

            flow_events = [event for event in trace_events
                           if event.get("ph") in (EventConstant.START_FLOW, EventConstant.END_FLOW)]

            # three pairs of flow events (framework-to-framework and framework-to-hardware and acl-to-hardware)
            self.assertEqual(len(flow_events), 6)

            # Verify flow event types
            flow_cats = {event["cat"] for event in flow_events}
            self.assertIn(EventConstant.MINDSPORE_SELF_FLOW_CAT, flow_cats)
            self.assertIn(EventConstant.MINDSPORE_SELF_FLOW_CAT, flow_cats)

    def test_assemble_scope_layer_events_should_create_scope_hierarchy_when_events_exist(self):
        """Should create correct scope layer hierarchy when events exist."""
        with (mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                         side_effect=lambda x: Decimal(x) / Decimal('1000'))):
            data = {
                "mindspore_op_list": self.fwk_kbk_op_data,
                "cpu_op_lines": self.cpu_op_data,
                "msprof_timeline": self.msprof_data
            }
            self.assembler._assemble_basic_events(data)
            self.assembler._assemble_flow_events()
            self.assembler._assemble_scope_layer_events()

            container = self.assembler.trace_view_container
            scope_pool = container.get_pool_by_name(TimelineLayerName.SCOPER_LAYER.value)
            self.assertIsNotNone(scope_pool)

            events = scope_pool.get_complete_events()
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0].name, "Default")
            self.assertEqual(events[1].name, "network-TrainOneStepCell")

    def test_assemble_scope_layer_events_should_not_create_events_when_no_scope_data(self):
        """Should not create any scope layer events when no scope data exists."""
        with (mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                         side_effect=lambda x: Decimal(x) / Decimal('1000'))):
            data = {
                "mindspore_op_list": self.fwk_pynative_op_data,
                "msprof_timeline": [self.msprof_cann_data]
            }
            self.assembler._assemble_basic_events(data)

            self.assembler._assemble_scope_layer_events()

            container = self.assembler.trace_view_container
            scope_pool = container.get_pool_by_pid(TimelineLayerName.SCOPER_LAYER.value)
            self.assertIsNone(scope_pool)


if __name__ == '__main__':
    unittest.main()
