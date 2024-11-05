# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""Constant values"""
from enum import Enum


class DeviceTarget(Enum):
    """The device target enum."""
    CPU = 'cpu'
    GPU = 'gpu'
    ASCEND = 'ascend'


class ProfilerLevel(Enum):
    Level0 = "Level0"
    Level1 = "Level1"
    Level2 = "Level2"


class EventConstant:
    """Timeline event constant values"""

    START_FLOW = "s"
    END_FLOW = "f"
    META_EVENT = 'M'
    COMPLETE_EVENT = 'X'
    INSTANT_EVENT = 'i'
    COUNTER_EVENT = 'C'

    PROCESS_NAME = "process_name"
    PROCESS_LABEL = "process_labels"
    PROCESS_SORT = "process_sort_index"
    THREAD_NAME = "thread_name"
    THREAD_SORT = "thread_sort_index"

    HOST_TO_DEVICE_FLOW_CAT = "HostToDevice"
    MINDSPORE_NPU_FLOW_CAT = "async_npu"
    MINDSPORE_SELF_FLOW_CAT = "async_mindspore"

    MINDSPORE_PID = 1
    CPU_OP_PID = 2
    SCOPE_LAYER_PID = 3

    MINDSPORE_SORT_IDX = 1
    CPU_OP_SORT_IDX = 2
    SCOPE_LAYER_SORT_IDX = 12


class TimeConstant:
    """Time constant values"""

    NS_TO_US = 0.001
    MS_TO_US = 1000
