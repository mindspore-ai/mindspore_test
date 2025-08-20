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

"""Dataset module."""

from typing import Generic, Iterable, Iterator, TypeVar

from mindspore import Tensor

_T_co = TypeVar("_T_co", covariant=True)

class Dataset(Generic[_T_co]):
    """
    Base class for map style datasets.

    Map style datasets are datasets that represent a mapping from keys to data samples.
    Subclasses must overwrite `__getitem__` method, defining how to retrieve the samples
    according to the key. Subclasses could optionally overwrite `__len__` method, returning
    the size of the dataset.
    """
    def __init__(self) -> None:
        pass

    def __getitem__(self, index: int) -> _T_co:
        raise NotImplementedError(f"{self.__class__.__name__} should implement `__getitem__` method.")


class IterableDataset(Dataset[_T_co], Iterable[_T_co]):
    """
    Base class for iterable datasets.

    Iterable datasets are datasets that represent an iterator over data samples. It is particularly useful
    when random reads are expensive or even improbable. Subclasses must overwrite `__iter__` method, returning
    an iterator of samples over the dataset.
    """

    def __iter__(self) -> Iterator[_T_co]:
        raise NotImplementedError(f"{self.__class__.__name__} should implement `__iter__` method.")


class TensorDataset(Dataset[tuple[Tensor, ...]]):
    """
    Each sample is retrieved by indexing the input tensors along their first dimension.

    Args:
        *tensors (mindspore.Tensor): Input tensors. All tensors must have the same size in the first dimension.
    """
    def __init__(self, *tensors: Tensor) -> None:
        assert all(
            tensors[0].size == tensor.size for tensor in tensors
        ), "Size mismatch between tensors"
        super().__init__()
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size
