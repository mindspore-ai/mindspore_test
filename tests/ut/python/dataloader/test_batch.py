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


def test_mapdataset_batch():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    
    dataset = MyDataset(10)
    
    dataloader = ds.DataLoader(dataset, batch_size=1)
    compare_tensor_list(list(dataloader), [ms.Tensor([i]) for i in range(10)])
    
    dataloader = ds.DataLoader(dataset, batch_size=4, drop_last=False)
    result = list(dataloader)
    expect = [ms.Tensor([0, 1, 2, 3]), ms.Tensor([4, 5, 6, 7]), ms.Tensor([8, 9])]
    compare_tensor_list(result, expect)

    dataloader = ds.DataLoader(dataset, batch_size=4, drop_last=True)
    result = list(dataloader)
    expect = expect[:2]
    compare_tensor_list(result, expect)
    

def test_iterdataset_batch():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyIterDataset(10)
    dataloader = ds.DataLoader(dataset, batch_size=1)
    compare_tensor_list(list(dataloader), [ms.Tensor([i]) for i in range(10)])
    
    dataloader = ds.DataLoader(dataset, batch_size=4, drop_last=False)
    result = list(dataloader)
    expect = [ms.Tensor([0, 1, 2, 3]), ms.Tensor([4, 5, 6, 7]), ms.Tensor([8, 9])]
    compare_tensor_list(result, expect)

    dataloader = ds.DataLoader(dataset, batch_size=4, drop_last=True)
    result = list(dataloader)
    expect = expect[:2]
    compare_tensor_list(result, expect)


def test_mapdataset_batch_shuffle():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyDataset(10)
    ms.set_seed(0)
    dataloader = ds.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=False)
    compare_tensor_list([t.asnumpy() for t in list(dataloader)], [0, 2, 1, 5, 9, 8 ,4, 7, 6, 3])

    ms.set_seed(1)
    dataloader = ds.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=True)
    compare_tensor_list([t.asnumpy() for t in list(dataloader)], [9, 0, 2, 5, 7, 4, 6, 3, 1, 8])


def test_iterdataset_batch_shuffle():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyIterDataset(10)

    with pytest.raises(ValueError) as raise_info:
        dataloader = ds.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=False)
        for data in dataloader:
            print(data, type(data))
    assert "expected unspecified shuffle option, but got shuffle=True" in str(raise_info.value)