# SPDX-License-Identifier: Apache-2.0
# Standard
import abc
import json
import time
import io
from typing import Any
# Third Party
import torch


class Serializer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_bytes(self, t: torch.Tensor) -> bytes:
        """
        Serialize a pytorch tensor to bytes. The serialized bytes should contain
        both the data and the metadata (shape, dtype, etc.) of the tensor.

        Input:
            t: the input pytorch tensor, can be on any device, in any shape,
               with any dtype

        Returns:
            bytes: the serialized bytes
        """
        raise NotImplementedError


class SerializerDebugWrapper(Serializer):
    def __init__(self, s: Serializer):
        self.s = s

    def to_bytes(self, t: torch.Tensor) -> bytes:
        start = time.perf_counter()
        bs = self.s.to_bytes(t)
        end = time.perf_counter()

        # logger.debug(f"Serialization took {end - start:.2f} seconds")
        return bs


class Deserializer(metaclass=abc.ABCMeta):
    def __init__(self, dtype):
        self.dtype = dtype

    @abc.abstractmethod
    def from_bytes(self, bs: bytes) -> torch.Tensor:
        """
        Deserialize a pytorch tensor from bytes.

        Input:
            bytes: a stream of bytes

        Output:
            torch.Tensor: the deserialized pytorch tensor
        """
        raise NotImplementedError


class DeserializerDebugWrapper(Deserializer):
    def __init__(self, d: Deserializer):
        self.d = d

    def from_bytes(self, t: bytes) -> torch.Tensor:
        start = time.perf_counter()
        ret = self.d.from_bytes(t)
        end = time.perf_counter()

        # logger.debug(f"Deserialization took {(end - start) * 1000:.2f} ms")
        return ret


# class FastSerializer(Serializer):
#     def __init__(self):
#         super().__init__()

#     def to_bytes(self, t: torch.Tensor) -> bytes:
#         # make tensor into bit stream
#         buf = t.contiguous().cpu().view(torch.uint8).numpy().tobytes()
#         return buf


# class FastDeserializer(Deserializer):
#     def __init__(self, dtype):
#         super().__init__(dtype)

#     def from_bytes_normal(self, b: bytes) -> torch.Tensor:
#         print(self.dtype)
#         return torch.frombuffer(b, dtype=self.dtype)

#     def from_bytes(self, b: bytes) -> torch.Tensor:
#         return self.from_bytes_normal(b)


class TorchSerializer(Serializer):
    def __init__(self):
        super().__init__()

    def to_bytes(self, t: torch.Tensor) -> bytes:
        with io.BytesIO() as f:
            torch.save(t.cpu().clone().detach(), f)
            return f.getvalue()


class TorchDeserializer(Deserializer):
    def __init__(self, dtype):
        super().__init__(dtype)

    def from_bytes_normal(self, b: bytes) -> torch.Tensor:
        with io.BytesIO(b) as f:
            return torch.load(f)

    def from_bytes(self, b: bytes) -> torch.Tensor:
        return self.from_bytes_normal(b).to(dtype=self.dtype)


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


def dtype_to_name(dtype: torch.dtype) -> str:
    """Convert a torch.dtype to a short name suitable for RPC metadata."""
    return str(dtype).split(".")[-1]


def dtype_from_name(name: str) -> torch.dtype:
    """Convert a dtype string from metadata back into torch.dtype."""
    try:
        return getattr(torch, name)
    except AttributeError as exc:  # pragma: no cover - validation guard
        raise ValueError(f"Unknown tensor dtype '{name}'") from exc
