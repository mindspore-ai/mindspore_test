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
# ============================================================================"""
"""Framework event classes for timeline analysis."""
from enum import Enum
from decimal import Decimal
from typing import Dict, Optional, List, Tuple

from mindspore.profiler.common.constant import EventConstant, FileConstant
from mindspore.profiler.analysis.time_converter import TimeConverter
from mindspore.profiler.analysis.parser.timeline_event.base_event import (
    BaseEvent,
    CompleteEvent,
    MetaEvent,
    InstantEvent
)


class FwkFixSizeFormat:
    """Format definition for framework fixed-size data."""

    # Fixed size data format: 3 long long (q) + 6 unsigned long long (Q) + bool (?) + padding (b)
    OpRangeStruct = "<3q6Qb?"


class OpRangeStructField(Enum):
    """Field indices in operator range structure fixed-size data."""
    START_NS = 0
    END_NS = 1
    SEQUENCE_NUMBER = 2
    PROCESS_ID = 3
    START_THREAD_ID = 4
    END_THREAD_ID = 5
    FORWARD_THREAD_ID = 6
    ID = 7
    STEP_ID = 8
    LEVEL = 9
    IS_ASYNC = 10


class FwkArgsDecoder:
    """Decoder for framework event arguments in TLV (Type-Length-Value) format."""
    TLV_TYPES = {
        EventConstant.OP_NAME: 3,
        EventConstant.INPUT_SHAPES: 5,
        EventConstant.INPUT_DTYPES: 4,
        EventConstant.CALL_STACK: 6,
        EventConstant.MODULE_HIERARCHY: 7,
        EventConstant.FLOPS: 8,
        EventConstant.CUSTOM_INFO: 9
    }

    @classmethod
    def decode(cls, origin_data: Dict, fix_size_data: Tuple) -> Dict:
        """Decode event arguments from raw data."""
        args = {
            EventConstant.SEQUENCE_NUMBER: int(fix_size_data[OpRangeStructField.SEQUENCE_NUMBER.value]),
            EventConstant.FORWARD_THREAD_ID: int(fix_size_data[OpRangeStructField.FORWARD_THREAD_ID.value])
        }

        for field_name, type_id in cls.TLV_TYPES.items():
            if field_name == EventConstant.OP_NAME or type_id not in origin_data:
                continue
            value = origin_data.get(type_id)
            if field_name in {EventConstant.INPUT_SHAPES, EventConstant.INPUT_DTYPES, EventConstant.CALL_STACK}:
                args[field_name] = value.replace("|", "\r\n")
            elif field_name == EventConstant.CUSTOM_INFO and value:
                pairs = [pair.split(':') for pair in value.split(';') if pair]
                info_dict = {k: v for k, v in pairs[0:2] if len(pairs) >= 2}
                args[field_name] = info_dict
            else:
                args[field_name] = value

        return args


class FwkCompleteEvent(CompleteEvent):
    """Framework complete event with duration."""
    _args_decoder = FwkArgsDecoder()

    def __init__(self, data: Dict):
        """Initialize framework complete event."""
        super().__init__(data)
        self.fix_size_data = self._origin_data[FileConstant.FIX_SIZE_DATA]
        self._ts_cache = None
        self._te_cache = None
        self._dur_cache = None
        self._args_cache = {}
        self._parent: Optional[BaseEvent] = None
        self._children: List[BaseEvent] = []

    @property
    def parent(self) -> BaseEvent:
        """Get parent event."""
        return self._parent

    @parent.setter
    def parent(self, event: BaseEvent) -> None:
        """Set parent event."""
        self._parent = event

    @property
    def children(self) -> List[BaseEvent]:
        """Get child events."""
        return self._children

    @property
    def ts_raw(self) -> int:
        """Get raw start timestamp."""
        return self.fix_size_data[OpRangeStructField.START_NS.value]

    @property
    def ts(self) -> Decimal:
        """Get start time in us."""
        if not self._ts_cache:
            self._ts_cache = TimeConverter.convert_syscnt_to_timestamp_us(
                self.fix_size_data[OpRangeStructField.START_NS.value]
            )
        return self._ts_cache

    @property
    def te(self) -> Decimal:
        """Get end time in us."""
        if not self._te_cache:
            self._te_cache = TimeConverter.convert_syscnt_to_timestamp_us(
                self.fix_size_data[OpRangeStructField.END_NS.value]
            )
        return self._te_cache

    @property
    def dur(self) -> Decimal:
        """Get duration in us."""
        if not self._dur_cache:
            self._dur_cache = self.te - self.ts
        return self._dur_cache

    @property
    def pid(self) -> int:
        """Get process ID."""
        return int(EventConstant.MINDSPORE_PID)

    @property
    def tid(self) -> int:
        """Get thread ID."""
        return int(self.fix_size_data[OpRangeStructField.START_THREAD_ID.value])

    @property
    def id(self) -> int:
        """Get event ID."""
        return int(self.fix_size_data[OpRangeStructField.ID.value])

    @property
    def name(self) -> str:
        """Get operator name."""
        return str(self._origin_data.get(self._args_decoder.TLV_TYPES.get(EventConstant.OP_NAME), ""))

    @property
    def step(self) -> int:
        """Get step ID."""
        return int(self.fix_size_data[OpRangeStructField.STEP_ID.value])

    @property
    def level(self) -> int:
        """Get event level."""
        return int(self.fix_size_data[OpRangeStructField.LEVEL.value])

    @property
    def args(self) -> Dict:
        """Get decoded event arguments."""
        if not self._args_cache:
            self._args_cache = self._args_decoder.decode(self._origin_data, self.fix_size_data)
        return self._args_cache

    @property
    def custom_info(self) -> str:
        """Get custom information."""
        return str(self.args.get(EventConstant.CUSTOM_INFO, ''))


class FwkInstantEvent(InstantEvent):
    """Framework instant event without duration."""
    _args_decoder = FwkArgsDecoder()

    def __init__(self, data: Dict):
        """Initialize framework instant event."""
        super().__init__(data)
        self.fix_size_data = self._origin_data[FileConstant.FIX_SIZE_DATA]
        self._ts_cache = None
        self._args_cache = {}

    @property
    def ts_raw(self) -> int:
        """Get raw start timestamp."""
        return self.fix_size_data[OpRangeStructField.START_NS.value]

    @property
    def ts(self) -> Decimal:
        """Get time in us."""
        if not self._ts_cache:
            self._ts_cache = TimeConverter.convert_syscnt_to_timestamp_us(
                self.fix_size_data[OpRangeStructField.START_NS.value]
            )
        return self._ts_cache

    @property
    def pid(self) -> int:
        """Get process ID."""
        return int(EventConstant.MINDSPORE_PID)

    @property
    def tid(self) -> int:
        """Get thread ID."""
        return int(self.fix_size_data[OpRangeStructField.START_THREAD_ID.value])

    @property
    def name(self) -> str:
        """Get operator name."""
        return str(self._origin_data.get(self._args_decoder.TLV_TYPES.get(EventConstant.OP_NAME), ""))

    @property
    def step(self) -> int:
        """Get step ID."""
        return int(self.fix_size_data[OpRangeStructField.STEP_ID.value])

    @property
    def level(self) -> int:
        """Get event level."""
        return int(self.fix_size_data[OpRangeStructField.LEVEL.value])

    @property
    def args(self) -> Dict:
        """Get decoded event arguments."""
        if not self._args_cache:
            self._args_cache = self._args_decoder.decode(self._origin_data, self.fix_size_data)
        return self._args_cache

    @property
    def custom_info(self) -> str:
        """Get custom information."""
        return str(self.args.get(EventConstant.CUSTOM_INFO, ''))


class FwkMetaEvent(MetaEvent):
    """Framework metadata event."""

    @property
    def pid(self) -> int:
        """Get framework process ID."""
        return int(EventConstant.MINDSPORE_PID)
