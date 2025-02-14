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

from typing import Generic, Iterable, Iterator, TypeVar

_T_co = TypeVar("_T_co", covariant=True)

class Dataset(Generic[_T_co]):
    def __init__(self) -> None:
        pass

    def __getitem__(self, index):
        raise NotImplementedError("{} should implement `__getitem__` method.".format(self.__class__.__name__))

    def __len__(self):
        raise NotImplementedError("{} should implement `__len__` method.".format(self.__class__.__name__))


class IterableDataset(Dataset[_T_co], Iterable[_T_co]):
    def __init__(self) -> None:
        pass

    def __iter__(self):
        raise NotImplementedError("{} should implement `__iter__` method.".format(self.__class__.__name__))