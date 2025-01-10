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

from mindspore.profiler.common.constant import EventConstant, FileConstant
from mindspore.profiler.analysis.parser.timeline_event.fwk_event import (
    FwkCompleteEvent,
    FwkInstantEvent,
    FwkMetaEvent,
    FwkArgsDecoder
)


class TestFwkArgsDecoder(unittest.TestCase):
    """Test cases for Framework arguments decoder."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.fix_size_data = [0, 0, 1, 1, 2, 2, 3, 123, 0, 1, False]
        self.origin_data = {
            4: "float32",  # INPUT_DTYPES
            5: "1|2|3",    # INPUT_SHAPES
            6: "a|b|c",    # CALL_STACK
            7: "hierarchy",  # MODULE_HIERARCHY
            8: "100",      # FLOPS
            9: "key1:value1;key2:value2"  # CUSTOM_INFO
        }

    def test_decoder_should_process_sequence_and_thread_id_when_fix_size_data_provided(self):
        """Should process sequence number and forward thread ID when fix size data is provided."""
        args = FwkArgsDecoder.decode(self.origin_data, self.fix_size_data)
        self.assertEqual(args[EventConstant.SEQUENCE_NUMBER], 1)
        self.assertEqual(args[EventConstant.FORWARD_THREAD_ID], 3)

    def test_decoder_should_convert_separator_when_list_type_data_provided(self):
        """Should convert separator from | to \r\n when processing list type data."""
        args = FwkArgsDecoder.decode(self.origin_data, self.fix_size_data)
        self.assertEqual(args[EventConstant.INPUT_DTYPES], "float32")
        self.assertEqual(args[EventConstant.INPUT_SHAPES], "1\r\n2\r\n3")
        self.assertEqual(args[EventConstant.CALL_STACK], "a\r\nb\r\nc")

    def test_decoder_should_parse_custom_info_when_key_value_pairs_provided(self):
        """Should parse custom info into dictionary when key-value pairs are provided."""
        args = FwkArgsDecoder.decode(self.origin_data, self.fix_size_data)
        self.assertIsInstance(args[EventConstant.CUSTOM_INFO], dict)
        self.assertEqual(args[EventConstant.CUSTOM_INFO]["key1"], "value1")
        self.assertEqual(args[EventConstant.CUSTOM_INFO]["key2"], "value2")


class TestFwkEvent(unittest.TestCase):
    """Test cases for Framework event classes."""

    def setUp(self):
        """Set up test environment before each test case."""
        self.fix_size_data = (
            1000000,  # START_NS
            2000000,  # END_NS
            1,        # SEQUENCE_NUMBER
            1,        # PROCESS_ID
            2,        # START_THREAD_ID
            2,        # END_THREAD_ID
            3,        # FORWARD_THREAD_ID
            123,      # ID
            0,        # STEP_ID
            1,        # LEVEL
            False     # IS_ASYNC
        )
        self.event_data = {
            FileConstant.FIX_SIZE_DATA: self.fix_size_data,
            3: "test_op",  # OP_NAME
        }
        self.mock_args = {
            EventConstant.SEQUENCE_NUMBER: 1,
            EventConstant.FORWARD_THREAD_ID: 3,
            EventConstant.CUSTOM_INFO: {"key1": "value1"}
        }

    def test_complete_event_properties_should_return_correct_values_when_initialized(self):
        """Should return correct values for all complete event properties when event is initialized."""
        with mock.patch.object(FwkArgsDecoder, 'decode', return_value=self.mock_args):
            with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                            side_effect=lambda x: Decimal(x) / Decimal('1000')):
                event = FwkCompleteEvent(self.event_data)
                self.assertEqual(event.ts_raw, 1000000)
                self.assertEqual(event.ts, Decimal('1000'))
                self.assertEqual(event.te, Decimal('2000'))
                self.assertEqual(event.dur, Decimal('1000'))
                self.assertEqual(event.pid, EventConstant.MINDSPORE_PID)
                self.assertEqual(event.tid, 2)
                self.assertEqual(event.id, 123)
                self.assertEqual(event.name, "test_op")
                self.assertEqual(event.step, 0)
                self.assertEqual(event.level, 1)
                self.assertEqual(event.ph, EventConstant.COMPLETE_EVENT)
                self.assertIsNone(event.parent)
                self.assertEqual(event.children, [])

    def test_instant_event_properties_should_return_correct_values_when_initialized(self):
        """Should return correct values for all instant event properties when event is initialized."""
        with mock.patch.object(FwkArgsDecoder, 'decode', return_value=self.mock_args):
            with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                            side_effect=lambda x: Decimal(x) / Decimal('1000')):
                event = FwkInstantEvent(self.event_data)
                self.assertEqual(event.ts_raw, 1000000)
                self.assertEqual(event.ts, Decimal('1000'))
                self.assertEqual(event.pid, EventConstant.MINDSPORE_PID)
                self.assertEqual(event.tid, 2)
                self.assertEqual(event.name, "test_op")
                self.assertEqual(event.step, 0)
                self.assertEqual(event.level, 1)
                self.assertEqual(event.ph, EventConstant.INSTANT_EVENT)

    def test_meta_event_properties_should_return_correct_values_when_initialized(self):
        """Should return correct values for all meta event properties when event is initialized."""
        meta_data = {
            "name": "process_name",
            "args": {"name": "test_process"}
        }
        event = FwkMetaEvent(meta_data)
        self.assertEqual(event.pid, EventConstant.MINDSPORE_PID)
        self.assertEqual(event.name, "process_name")
        self.assertEqual(event.args["name"], "test_process")
        self.assertEqual(event.ph, EventConstant.META_EVENT)

    def test_complete_event_should_return_default_values_when_fields_not_exist(self):
        """Return default values when fields do not exist."""
        event_data = {
            FileConstant.FIX_SIZE_DATA: self.fix_size_data,
            # No OP_NAME field
        }
        with mock.patch.object(FwkArgsDecoder, 'decode', return_value={}):
            with mock.patch("mindspore.profiler.analysis.time_converter.TimeConverter.convert_syscnt_to_timestamp_us",
                            side_effect=lambda x: Decimal(x) / Decimal('1000')):
                event = FwkCompleteEvent(event_data)
                self.assertEqual(event.name, "")  # Default empty string for name
                self.assertEqual(event.custom_info, "")  # Default empty string for custom_info
                self.assertEqual(event.args, {})  # Default empty dict for args


if __name__ == '__main__':
    unittest.main()
