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

from typing import Generic, Iterable, Iterator, TypeVar, Union

_T_co = TypeVar("_T_co", covariant=True)


class Sampler(Generic[_T_co]):
    def __init__(self, data_source) -> None:
        pass

class SequentialSampler(Sampler):
    def __init__(self, data_source) -> None:
        self.data_source = data_source

    def __iter__(self)->Iterator[int]:
        yield from range(len(self.data_source))

    def __len__(self):
        return len(self.data_source)


class RandomSampler(Sampler):
    def __init__(self,
                 data_source,
                 replacement: bool=False,
                 num_samples: Union[int, None] = None,
                 generator=None) -> None:
        pass

class BatachSampler(Sampler):
    def __init__(self,
                 sampler: Union[Sampler, Iterable],
                 batch_size: int,
                 drop_last: bool) -> None:
        pass