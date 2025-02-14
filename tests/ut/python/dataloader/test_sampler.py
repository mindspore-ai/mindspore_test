import collections
import numpy as np
import pytest

import mindspore as ms
import mindspore.dataset.dataloader as ds


class MyDataset(ds.Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = [idx for idx in range(num_samples)]

    def __getitem__(self, index):
        return np.array(self.data[index])

    def __len__(self):
        return self.num_samples
    

class MyIterDataset(ds.dataset.IterableDataset):
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
        else:
            raise StopIteration
    

def compare_tensor_list(list1, list2):
    assert len(list1) == len(list2)
    for v1, v2 in zip(list1, list2):
        assert (v1 == v2).all()


def test_dataloader_udf_sampler():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyDataset(10)
    sampler = MySampler(6)
    
    dataloader = ds.DataLoader(dataset, batch_size=2, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor([i, i + 1]) for i in range(0, 6, 2)]
    compare_tensor_list(result, expect)


def test_dataloader_sequential_sampler():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyDataset(10)
    
    dataloader1 = ds.DataLoader(dataset, batch_size=1, sampler=ds.sampler.SequentialSampler(dataset), shuffle=False)
    result1 = list(dataloader1)

    dataloader2 = ds.DataLoader(dataset, batch_size=1, sampler=None, shuffle=False)
    result2 = list(dataloader2)

    compare_tensor_list(result1, result2)


def test_dataloader_random_sampler():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyDataset(10)
    
    # get_seed 默认返回None
    # ops.randint的seed参数可以接受None，可以计算
    sampler = ds.sampler.RandomSampler(dataset, replacement=True)
    dataloader = ds.DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)
    result = list(dataloader)
    print(result)

    # 指定num_samples
    sampler = ds.sampler.RandomSampler(dataset, replacement=True, num_samples=3)
    dataloader = ds.DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)
    result = list(dataloader)
    print(result)

    # 但ops.randperm的seed参数不接受None，导致random sampler报错
    sampler = ds.sampler.RandomSampler(dataset, replacement=False)
    dataloader = ds.DataLoader(dataset, batch_size=1, sampler=sampler, shuffle=False)
    result = list(dataloader)


def test_dataloader_batch_sampler():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyDataset(10)

    class SimpleBatchSampler:
        def __init__(self):
            self.indices = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        def __iter__(self):
            return iter(self.indices)
    
    dataloader = ds.DataLoader(dataset, batch_size=1, batch_sampler=SimpleBatchSampler(), shuffle=False)
    result = list(dataloader)
    expected = [ms.Tensor([i, i+1]) for i in range(0, 10, 2)]
    compare_tensor_list(result, expected)

    error_msg = "`batch_sampler` can not specify with `batch_size`, `drop_last`, `shuffle` or `sampler`"
    with pytest.raises(ValueError) as raise_info:
        dataloader = ds.DataLoader(dataset, batch_size=2, batch_sampler=SimpleBatchSampler(), shuffle=False)
        list(dataloader)
    assert error_msg in str(raise_info.value)
    
    with pytest.raises(ValueError) as raise_info:
        dataloader = ds.DataLoader(dataset, batch_size=1, batch_sampler=SimpleBatchSampler(), shuffle=True)
        list(dataloader)
    assert error_msg in str(raise_info.value)
    
    with pytest.raises(ValueError) as raise_info:
        dataloader = ds.DataLoader(dataset, batch_size=1, batch_sampler=SimpleBatchSampler(), drop_last=True)
        list(dataloader)
    assert error_msg in str(raise_info.value)


def test_dataloader_sampler_conflict():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyDataset(10)
    sampler = MySampler(6)
    
    dataloader = ds.DataLoader(dataset, sampler=sampler, shuffle=False)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(6)]
    compare_tensor_list(result, expect)
    
    with pytest.raises(ValueError) as raise_info:
        dataloader = ds.DataLoader(dataset, sampler=sampler, shuffle=True)
        result = list(dataloader)
    assert "`shuffle` and `sampler` can not specify at the same time" in str(raise_info.value)


def test_dataloader_distributed_sampler():
    """
    Feature:
    Description:
    Expectation:
    """
    dataset = MyDataset(10)

    sampler = ds.distributed.DistributedSampler(dataset, shuffle=False,
                                                num_replicas=None, rank=None)
    dataloader = ds.DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(0, 10)]
    compare_tensor_list(result, expect)
    print(result)

    sampler = ds.distributed.DistributedSampler(dataset, shuffle=False,
                                                num_replicas=2, rank=0)
    dataloader = ds.DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(0, 10, 2)]
    compare_tensor_list(result, expect)
    print(result)

    sampler = ds.distributed.DistributedSampler(dataset, shuffle=False,
                                                num_replicas=2, rank=1)
    dataloader = ds.DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in range(1, 10, 2)]
    compare_tensor_list(result, expect)
    print(result)


def test_dataloader_distributed_sampler_shuffle():
    """
    Feature:
    Description:
    Expectation:
    """
    dataset = MyDataset(10)

    sampler = ds.distributed.DistributedSampler(dataset, shuffle=True, seed=1,
                                                num_replicas=2, rank=0)
    dataloader = ds.DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [9, 2, 7, 6, 1]]
    compare_tensor_list(result, expect)
    print(result)


def test_dataloader_distributed_sampler_drop_last():
    """
    Feature:
    Description:
    Expectation:
    """
    dataset = MyDataset(10)
    '''
    sampler = ds.distributed.DistributedSampler(dataset, shuffle=False,
                                                num_replicas=3, rank=2, drop_last=False)
    dataloader = ds.DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [2, 5, 8, 1]]
    compare_tensor_list(result, expect)
    print(result)

    sampler = ds.distributed.DistributedSampler(dataset, shuffle=False,
                                                num_replicas=3, rank=2, drop_last=True)
    dataloader = ds.DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    expect = [ms.Tensor(i) for i in [2, 5, 8]]
    compare_tensor_list(result, expect)
    print(result)
    '''

    sampler = ds.distributed.DistributedSampler(dataset, shuffle=False,
                                                num_replicas=20, rank=0, drop_last=False)
    dataloader = ds.DataLoader(dataset, batch_size=None, sampler=sampler)
    result = list(dataloader)
    #expect = [ms.Tensor(i) for i in [2, 5, 8]]
    #compare_tensor_list(result, expect)
    print(result)



def test_dataloader_distributed_sampler_exception():
    """
    Feature:
    Description:
    Expectation:
    """
    dataset = MyDataset(10)

    error_msg = "num_replicas should be greater than 0."
    with pytest.raises(ValueError) as raise_info:
        _ = ds.distributed.DistributedSampler(dataset, shuffle=False, num_replicas=0, rank=0)
    assert error_msg in str(raise_info.value)

    error_msg = "rank should be in the interval"
    with pytest.raises(ValueError) as raise_info:
        _ = ds.distributed.DistributedSampler(dataset, shuffle=False, num_replicas=2, rank=2)
    assert error_msg in str(raise_info.value)
