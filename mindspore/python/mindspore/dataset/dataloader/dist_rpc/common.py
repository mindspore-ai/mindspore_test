# Copyright 2025 Huawei Technologies Co., Ltd
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
# ==============================================================================
"""Shared enums and constants that describe the distributed RPC protocol."""

from enum import Enum, IntEnum
import struct
from typing import Final


class PayloadType(IntEnum):
    JSON = 0
    BYTES = 1
    TENSOR = 2


class RPCMethod(str, Enum):
    REGISTER_CLIENT = "register_client"
    ASSIGN_SERVERNODE = "assign_servernode"
    FETCH = "fetch"
    REGISTER_SERVERNODE = "register_servernode" 
    REPORT_COMPLETION = "report_completion"     


REQUEST_HEADER: Final[struct.Struct] = struct.Struct("!III")
RESPONSE_HEADER: Final[struct.Struct] = struct.Struct("!IIII")
