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
"""Serialization helpers that convert RPC payloads to and from bytes."""

import abc
import io
import json
from typing import Any

import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype


class Serializer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_bytes(self, tensor: Tensor) -> bytes:
        """Serialize a MindSpore tensor (any shape/dtype) to bytes."""


class SerializerDebugWrapper(Serializer):
    def __init__(self, serializer: Serializer):
        self.serializer = serializer

    def to_bytes(self, tensor: Tensor) -> bytes:
        return self.serializer.to_bytes(tensor)


class Deserializer(metaclass=abc.ABCMeta):
    def __init__(self, dtype: Any):
        self.dtype = dtype

    @abc.abstractmethod
    def from_bytes(self, payload: bytes) -> Tensor:
        """Deserialize bytes into a MindSpore tensor."""


class DeserializerDebugWrapper(Deserializer):
    def __init__(self, deserializer: Deserializer):
        super().__init__(deserializer.dtype)
        self.deserializer = deserializer

    def from_bytes(self, payload: bytes) -> Tensor:
        return self.deserializer.from_bytes(payload)


class MindSporeSerializer(Serializer):
    """Serialize tensors via NumPy so both data and metadata are preserved."""

    def to_bytes(self, tensor: Tensor) -> bytes:
        with io.BytesIO() as buffer:
            np.save(buffer, tensor.asnumpy(), allow_pickle=False)
            return buffer.getvalue()


class MindSporeDeserializer(Deserializer):
    """Deserialize tensors written by :class:`MindSporeSerializer`."""

    def __init__(self, dtype: Any):
        super().__init__(dtype)

    def from_bytes_normal(self, payload: bytes) -> np.ndarray:
        with io.BytesIO(payload) as buffer:
            return np.load(buffer, allow_pickle=False)

    def from_bytes(self, payload: bytes) -> Tensor:
        array = self.from_bytes_normal(payload)
        tensor = Tensor(array)
        return tensor.astype(self.dtype) if self.dtype is not None else tensor


def serialize_message(payload: Any) -> bytes:
    """Serialize small control-plane payloads using JSON."""
    if payload is None:
        return b""
    return json.dumps(payload).encode("utf-8")


def deserialize_message(data: bytes) -> Any:
    """Deserialize JSON control-plane payloads."""
    if not data:
        return None
    return json.loads(data.decode("utf-8"))


def dtype_to_name(dtype: Any) -> str:
    """Convert a MindSpore dtype into a short name for RPC metadata."""
    if hasattr(dtype, "name"):
        return dtype.name
    return str(dtype)


def dtype_from_name(name: str) -> Any:
    """Convert a dtype string from metadata back into a MindSpore dtype."""
    candidates = {name, name.lower(), name.upper(), name.capitalize()}
    for candidate in candidates:
        dtype = getattr(mstype, candidate, None)
        if dtype is not None:
            return dtype
    raise ValueError(f"Unknown tensor dtype '{name}'")
