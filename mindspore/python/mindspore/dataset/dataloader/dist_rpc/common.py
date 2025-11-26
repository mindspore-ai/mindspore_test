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


REQUEST_HEADER: Final[struct.Struct] = struct.Struct("!III")
RESPONSE_HEADER: Final[struct.Struct] = struct.Struct("!IIII")
