from mindspore.dataset.dataloader.utils.collate import default_convert, default_collate
import mindspore.dataset as ds
import numpy as np
import mindspore as ms
import collections

import pytest

def test1():
    assert default_convert(1) == 1
    assert default_convert(np.array([1])) == ms.Tensor([1])

def test2():
    res1 = default_convert([np.array([0,1]), np.array([2,3]), 5])
    exp1 = [ms.Tensor([0,1]), ms.Tensor([2,3]), 5]

    for t, e in zip(res1, exp1):
        if isinstance(t, ms.Tensor):
            assert (t == e).all()
        else:
            assert (t == e)


class ImmutableMapping(collections.abc.Mapping):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

class UnsupportedMutableMapping(collections.abc.MutableMapping):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __setitem__(self, key, value):
        self._data[key] = value

    def __delitem__(self, key):
        del self._data[key]

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)

    # 故意不实现 copy 方法
    def __copy__(self):
        raise TypeError("不支持 copy 操作")

def test3():
    dataloader1 = default_convert(ImmutableMapping({0: "a", 1: "b"}))
    print(dataloader1)
    dataloader2 = default_convert(UnsupportedMutableMapping({0: "x", 1: "y"}))
    assert (dataloader2 == {0: 'x', 1: 'y'})


def test4():
    inputs = [ms.Tensor(np.array(1, dtype=np.uint8)), ms.Tensor(np.array(False, dtype=np.uint8))]
    # inputs = [ms.Tensor(np.array(1, dtype=np.uint8)), ms.Tensor(np.array(False, dtype=np.float32))]
    print(type(default_collate(inputs)))



def test_dataloader():
    dataset = ds.Dataset()
    dataloader = ds.dataloader.DataLoader(dataset)


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


class MyDataset(ds.Dataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = [idx for idx in range(num_samples)]

    def __getitem__(self, index):
        return np.array(self.data[index])

    def __len__(self):
        return self.num_samples
    

class MyIterDataset(ds.dataloader.dataset.IterableDataset):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = [np.array(idx) for idx in range(num_samples)]

    def __iter__(self):
        return iter(self.data)



def test_dataloader_single_process_iteration_mapdataset():
    dataset = MyDataset(10)
    sampler = MySampler(5)
    dataloader = ds.dataloader.DataLoader(dataset, batch_size=None, sampler=sampler)
    for data in dataloader:
        print(data, type(data))


def test_dataloader_single_process_iteration_iterdataset():
    dataset = MyIterDataset(10)
    sampler = MySampler(5)
    dataloader = ds.dataloader.DataLoader(dataset, batch_size=None)
    for data in dataloader:
        print(data, type(data))


def test_dataloader_single_process_iteration_iterdataset_batchsampler():
    dataset = MyIterDataset(10)
    #sampler = MySampler(5)
    dataloader = ds.dataloader.DataLoader(dataset, batch_size=3, drop_last=False)
    for data in dataloader:
        print(data, type(data))

    dataloader = ds.dataloader.DataLoader(dataset, batch_size=3, drop_last=True)
    for data in dataloader:
        print(data, type(data))

def test_dataloader_single_process_iteration_iterdataset_batchsampler2():
    dataset = MyIterDataset(10)
    #sampler = MySampler(5)
    dataloader = ds.dataloader.DataLoader(dataset, batch_size=1, drop_last=False)
    for data in dataloader:
        print(data, type(data))

test_dataloader_single_process_iteration_iterdataset_batchsampler2()

def test_dataloader_single_process_iteration_mapdataset_batchsampler():
    dataset = MyDataset(10)
    #sampler = MySampler(5)
    dataloader = ds.dataloader.DataLoader(dataset, batch_size=3, drop_last=False)
    for data in dataloader:
        print(data, type(data))

    dataloader = ds.dataloader.DataLoader(dataset, batch_size=3, drop_last=True)
    for data in dataloader:
        print(data, type(data))


def test_dataloader_single_process_iteration_mapdataset_batchsampler_shuffle():
    dataset = MyDataset(10)
    #sampler = MySampler(5)
    ms.set_seed(0)
    dataloader = ds.dataloader.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=False)
    for data in dataloader:
        print(data, type(data))

    ms.set_seed(1)
    dataloader = ds.dataloader.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=True)
    for data in dataloader:
        print(data, type(data))


def test_dataloader_single_process_iteration_iterdataset_batchsampler_shuffle():
    dataset = MyIterDataset(10)
    #sampler = MySampler(5)

    with pytest.raises(ValueError, match="DataLoader with IterableDataset: expected unspecified shuffle option, but got shuffle=True"):
        dataloader = ds.dataloader.DataLoader(dataset, batch_size=3, shuffle=True, drop_last=False)
        for data in dataloader:
            print(data, type(data))


def test_dataloader_single_process_iteration_mapdataset_sampler_shuffle():
    dataset = MyDataset(10)
    sampler = MySampler(5)

    with pytest.raises(ValueError, match="`shuffle` and `sampler` can not specify at the same time"):
        dataloader = ds.dataloader.DataLoader(dataset, batch_size=3, shuffle=True, sampler=sampler)
        for data in dataloader:
            print(data, type(data))

