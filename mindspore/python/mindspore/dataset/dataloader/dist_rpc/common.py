# SPDX-License-Identifier: Apache-2.0
"""Shared RPC protocol definitions."""

# Standard
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
    REGISTER_SERVERNODE = "register_servernode" # 顺便把节点注册也加上
    REPORT_COMPLETION = "report_completion"     # 新增：汇报完成情况


REQUEST_HEADER: Final[struct.Struct] = struct.Struct("!III")
RESPONSE_HEADER: Final[struct.Struct] = struct.Struct("!IIII")
