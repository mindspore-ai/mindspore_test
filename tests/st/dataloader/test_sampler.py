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

import numpy as np
import pytest

import mindspore as ms
from mindspore.dataset.dataloader import (
    DataLoader,
    Dataset,
    IterableDataset,
    SequentialSampler,
    RandomSampler,
    DistributedSampler,
)
from tests.mark_utils import arg_mark


class MyDataset(Dataset):

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = [idx for idx in range(num_samples)]

    def __getitem__(self, index):
        return np.array(self.data[index])

    def __len__(self):
        return self.num_samples


class MyIterDataset(IterableDataset):

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = [np.array(idx) for idx in range(num_samples)]

    def __iter__(self):
        return iter(self.data)


class MySampler:

    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index < self.num_samples:
            data = self.index
            self.index += 1
            return data
        raise StopIteration


def compare_tensor_list(list1, list2):
    assert len(list1) == len(list2)
    for v1, v2 in zip(list1, list2):
        assert (v1 == v2).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_udf_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with UDF sampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)
    sampler = MySampler(6)

    dataloader = DataLoader(dataset, batch_size=2, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor([i, i + 1]) for i in range(0, 6, 2)]
    compare_tensor_list(result, expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_sequential_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with SequentialSampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    dataloader1 = DataLoader(dataset, batch_size=1, sampler=SequentialSampler(dataset), shuffle=False)
    result1 = list(dataloader1)

    dataloader2 = DataLoader(dataset, batch_size=1, sampler=None, shuffle=False)
    result2 = list(dataloader2)

    compare_tensor_list(result1, result2)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="RandomSampler is not supported.")
def test_dataloader_random_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with RandomSampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    # get_seed returns None by default
    # ops.randint's seed parameter can accept None and can compute
    sampler = RandomSampler(dataset, replacement=True)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)
    result = list(dataloader)
    print(result)

    # define num_samples
    sampler = RandomSampler(dataset, replacement=True, num_samples=3)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)
    result = list(dataloader)
    print(result)

    # but ops.randperm's seed parameter does not accept None, causing random sampler to report an error
    sampler = RandomSampler(dataset, replacement=False)
    dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)
    result = list(dataloader)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_batch_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with batch_sampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    class SimpleBatchSampler:

        def __init__(self):
            self.indices = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        def __iter__(self):
            return iter(self.indices)

    dataloader = DataLoader(dataset, batch_size=1, batch_sampler=SimpleBatchSampler(), shuffle=False)
    result = list(dataloader)
    expected = [ms.Tensor([i, i + 1]) for i in range(0, 10, 2)]
    compare_tensor_list(result, expected)

    error_msg = ("`batch_sampler` can not specify with `batch_size`, `drop_last`, `shuffle` or `sampler`")
    with pytest.raises(ValueError) as raise_info:
        dataloader = DataLoader(dataset, batch_size=2, batch_sampler=SimpleBatchSampler(), shuffle=False)
        list(dataloader)
    assert error_msg in str(raise_info.value)

    with pytest.raises(ValueError) as raise_info:
        dataloader = DataLoader(dataset, batch_size=1, batch_sampler=SimpleBatchSampler(), shuffle=True)
        list(dataloader)
    assert error_msg in str(raise_info.value)

    with pytest.raises(ValueError) as raise_info:
        dataloader = DataLoader(dataset, batch_size=1, batch_sampler=SimpleBatchSampler(), drop_last=True)
        list(dataloader)
    assert error_msg in str(raise_info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_sampler_conflict():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader parameters conflict.
    Expectation: Raise ValueError.
    """

    dataset = MyDataset(10)
    sampler = MySampler(6)

    dataloader = DataLoader(dataset, sampler=sampler, shuffle=False)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(6)]
    compare_tensor_list(result, expect)

    with pytest.raises(ValueError) as raise_info:
        dataloader = DataLoader(dataset, sampler=sampler, shuffle=True)
        result = list(dataloader)
    assert "`shuffle` and `sampler` can not specify at the same time" in str(raise_info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_distributed_sampler():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with DistributedSampler.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=None, rank=None)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(0, 10)]
    compare_tensor_list(result, expect)
    print(result)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=2, rank=0)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(0, 10, 2)]
    compare_tensor_list(result, expect)
    print(result)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=2, rank=1)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(1, 10, 2)]
    compare_tensor_list(result, expect)
    print(result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_distributed_sampler_shuffle():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with DistributedSampler and shuffle.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    sampler = DistributedSampler(dataset, shuffle=True, seed=1, num_replicas=2, rank=0)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [9, 2, 7, 6, 1]]
    compare_tensor_list(result, expect)
    print(result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_distributed_sampler_drop_last():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with DistributedSampler and drop_last.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=3, rank=2, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [2, 5, 8, 1]]
    compare_tensor_list(result, expect)
    print(result)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=3, rank=2, drop_last=True)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [2, 5, 8]]
    compare_tensor_list(result, expect)
    print(result)

    sampler = DistributedSampler(dataset, shuffle=False, num_replicas=20, rank=0, drop_last=False)
    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [2, 5, 8]]
    print(result)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_distributed_sampler_exception():
    """
    Feature: Test DataLoader sampler.
    Description: Test the DataLoader with DistributedSampler parameters error.
    Expectation: Raise ValueError.
    """

    dataset = MyDataset(10)

    error_msg = "num_replicas should be greater than 0."
    with pytest.raises(ValueError) as raise_info:
        _ = DistributedSampler(dataset, shuffle=False, num_replicas=0, rank=0)
    assert error_msg in str(raise_info.value)

    error_msg = "rank should be in the interval"
    with pytest.raises(ValueError) as raise_info:
        _ = DistributedSampler(dataset, shuffle=False, num_replicas=2, rank=2)
    assert error_msg in str(raise_info.value)
