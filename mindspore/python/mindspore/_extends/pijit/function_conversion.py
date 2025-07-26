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
# ============================================================================
"""Convert the untraceable function into an equivalent one that pijit can trace."""

from collections.abc import Iterable, Iterator
import itertools
from typing import TypeVar

_T = TypeVar('_T')


# Reference: https://docs.python.org/3/library/itertools.html#itertools.chain
def itertools_chain(*iterables: Iterable[_T]) -> Iterator[_T]:
    # chain('ABC', 'DEF') → A B C D E F
    for iterable in iterables:
        yield from iterable


# Reference: https://docs.python.org/3/library/itertools.html#itertools.chain.from_iterable
def itertools_chain_from_iterable(cls: type, iterables: Iterable[Iterable[_T]]) -> Iterator[_T]:
    # chain.from_iterable(['ABC', 'DEF']) → A B C D E F
    for iterable in iterables:
        yield from iterable


# Convert the Python standard library API (usually implemented in C)
# into an equivalent function that is implemented entirely in Python,
# so that pijit can trace its bytecode.
_PIJIT_STDLIB_CONVERSION = {
    itertools.chain: itertools_chain,
    itertools.chain.from_iterable: itertools_chain_from_iterable
}
