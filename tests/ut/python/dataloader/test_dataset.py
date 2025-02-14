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

import pytest

import mindspore.dataset as ds


def test_iterate_dataset():
    dataset = ds.Dataset()
    with pytest.raises(NotImplementedError) as e:
        for _ in dataset:
            pass
    assert "Dataset should implement `__getitem__` method" in str(e.value)


def test_len_dataset():
    dataset = ds.Dataset()
    with pytest.raises(NotImplementedError) as e:
        _ = len(dataset)
    assert "Dataset should implement `__len__` method" in str(e.value)


def test_iterate_iterable_dataset():
    dataset = ds.IterableDataset()
    with pytest.raises(NotImplementedError) as e:
        for _ in dataset:
            pass
    assert "IterableDataset should implement `__iter__` method" in str(e.value)


def test_len_iterable_dataset():
    dataset = ds.IterableDataset()
    with pytest.raises(NotImplementedError) as e:
        _ = len(dataset)
    assert "IterableDataset should implement `__len__` method" in str(e.value)
