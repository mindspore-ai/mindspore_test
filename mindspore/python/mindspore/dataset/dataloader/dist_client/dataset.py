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

"""Utility datasets that feed the distributed dataloader demos."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import mindspore as ms
from mindspore import Tensor, ops


class DummyTextDataset:
    """Tiny text dataset that emits synthetic MindSpore tensors."""

    def __init__(self, size: int, seq_length: int) -> None:
        self.size = size
        self.seq_length = seq_length
        self.vocab_size = 32768

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> List[Dict[str, Tensor]]:  # pragma: no cover - simple demo
        _ = index
        input_ids = ops.randint(low=0, high=self.vocab_size, size=(self.seq_length,), dtype=ms.int32)
        attention_mask = ops.ones((self.seq_length,), ms.int32)
        labels = input_ids.copy()
        return [{"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}]


class MappingDataset:
    """Dataset wrapper that optionally applies a transform per sample."""

    def __init__(self, data: DummyTextDataset, transform: Optional[Callable] = None) -> None:
        self._data = data
        self._transform = transform

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, index: int) -> List[Dict[str, Tensor]]:  # pragma: no cover - simple demo
        if self._transform is not None:
            return self._transform(self._data[index])
        return self._data[index]


def build_dummy_dataset(task_type: str, size: int, max_seq_len: int) -> DummyTextDataset:
    """Factory that emits demo datasets by task type."""

    if task_type == "text":
        return DummyTextDataset(size=size, seq_length=max_seq_len)
    raise ValueError(f"Dummy dataset type ({task_type}) is not supported.")


def build_dummy_mapping_dataset(transform: Optional[Callable] = None) -> MappingDataset:
    """Build a mapping dataset using synthetic text samples."""

    size = 1000
    seq_length = 128
    dataset = DummyTextDataset(size=size, seq_length=seq_length)
    return MappingDataset(data=dataset, transform=transform)

