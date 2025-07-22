import collections
import multiprocessing
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
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.num_samples:
            data = self.index
            self.index += 1
            return data
        else:
            raise StopIteration
    

def test_dataloader_mapdataset_single_process():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyDataset(10)
    sampler = MySampler(5)
    
    dataloader = ds.DataLoader(dataset, batch_size=None)
    assert list(dataloader) == [ms.Tensor(i) for i in range(10)]
    
    dataloader = ds.DataLoader(dataset, batch_size=None, sampler=sampler)
    assert list(dataloader) == [ms.Tensor(i) for i in range(5)]


def test_dataloader_iterdataset_single_process():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyIterDataset(10)
    dataloader = ds.DataLoader(dataset, batch_size=None)
    assert list(dataloader) == [ms.Tensor(i) for i in range(10)]


def test_dataloader_mapdataset_multi_process():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    dataset = MyDataset(10)
    
    
    dataloader = ds.DataLoader(dataset, batch_size=3, num_workers=4, prefetch_factor=1)
    for data in dataloader:
        print(data)

    dataloader = ds.DataLoader(dataset, batch_size=1, num_workers=2, sampler=ds.sampler.RandomSampler(dataset, replacement=True))
    for data in dataloader:
        print(data)
    '''
    dataloader = ds.DataLoader(dataset, batch_size=2, num_workers=12, shuffle=False)
    for data in dataloader:
        print(data)
    '''
    

def test_dataloader_mapdataset_multi_process_exception():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    class ExceptionDataset(ds.Dataset):
        def __init__(self, num_samples):
            super().__init__()
            self.num_samples = num_samples
            self.data = [idx for idx in range(num_samples)]

        def __getitem__(self, index):
            if index == int(self.num_samples / 2):
                raise RuntimeError("I got an exception!!!")
            return np.array(self.data[index])

        def __len__(self):
            return self.num_samples

    dataset = ExceptionDataset(9)
    
    dataloader = ds.DataLoader(dataset, batch_size=3, num_workers=4, prefetch_factor=1)
    for data in dataloader:
        print(data)


def test_dataloader_iterdataset_multi_process():
    """
    Feature: 
    Description: 
    Expectation: 
    """
    '''
    import torch.utils.data as torchdata
    class MyIterDataset(torchdata.IterableDataset):
        def __init__(self, num_samples):
            super().__init__()
            self.num_samples = num_samples
            self.data = [np.array(idx) for idx in range(num_samples)]

        def __iter__(self):
            return iter(self.data)
    '''
    dataset = MyIterDataset(3)
    
    dataloader = ds.DataLoader(dataset, batch_size=None, num_workers=2, prefetch_factor=2)
    for data in dataloader:
        print(data)


def test_tensordataset():
    dataset = ds.TensorDataset(ms.Tensor([1, 2, 3, 4, 5]))
    print(len(dataset))
    for data in dataset:
        print(data)


def test_dataloader_iterdataset_multi_process_with_start_method():
    """
    Feature:
    Description:
    Expectation:
    """
    '''
    import torch.utils.data as torchdata
    class MyIterDataset(torchdata.IterableDataset):
        def __init__(self, num_samples):
            super().__init__()
            self.num_samples = num_samples
            self.data = [np.array(idx) for idx in range(num_samples)]

        def __iter__(self):
            return iter(self.data)
    '''
    dataset = MyIterDataset(3)

    dataloader = ds.DataLoader(dataset, batch_size=None, num_workers=2, prefetch_factor=2,
                               multiprocessing_context="fork")
    for data in dataloader:
        print(data)

    dataloader = ds.DataLoader(dataset, batch_size=None, num_workers=2, prefetch_factor=2,
                               multiprocessing_context=multiprocessing.get_context("spawn"))
    for data in dataloader:
        print(data)


def test_dataloader_iterdataset_multi_process_with_start_method_exception():
    """
    Feature:
    Description: Testing start_method exceptions under multiprocessing
    Expectation:
    """
    '''
    import torch.utils.data as torchdata
    class MyIterDataset(torchdata.IterableDataset):
        def __init__(self, num_samples):
            super().__init__()
            self.num_samples = num_samples
            self.data = [np.array(idx) for idx in range(num_samples)]

        def __iter__(self):
            return iter(self.data)
    '''
    dataset = MyIterDataset(3)

    with pytest.raises(ValueError) as e:
        dataloader = ds.DataLoader(dataset, batch_size=None, num_workers=2, prefetch_factor=2,
                                   multiprocessing_context="error_start_method")
        for _ in dataloader:
            pass
    assert "multiprocessing_context option should specify a valid start method in" in str(e.value)
