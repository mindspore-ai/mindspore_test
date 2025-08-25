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
"""Test DataLoader."""

import multiprocessing
import os
import random
import signal
import subprocess
import time

import numpy as np
import psutil
import pytest

import mindspore as ms
from mindspore.dataset.dataloader import (
    DataLoader,
    Dataset,
    default_collate,
    get_worker_info,
    IterableDataset,
    RandomSampler,
    TensorDataset,
)
from tests.mark_utils import arg_mark


class MyDataset(Dataset):
    """
    A map style dataset that returns as many samples as requested.
    """

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = [idx for idx in range(num_samples)]

    def __getitem__(self, index):
        return np.array(self.data[index])

    def __len__(self):
        return self.num_samples


class MyIterDataset(IterableDataset):
    """
    An iterable style dataset that yields as many samples as requested.
    """

    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.data = [np.array(idx) for idx in range(num_samples)]

    def __iter__(self):
        return iter(self.data)


class MySampler:
    """
    A sampler that yields as many indices as requested sequentially.
    """

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
        raise StopIteration


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_mapdataset_single_process():
    """
    Feature: Test DataLoader with MapDataset.
    Description: Test the DataLoader with MapDataset in single process.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)
    sampler = MySampler(5)

    dataloader = DataLoader(dataset, batch_size=None)
    assert list(dataloader) == [ms.Tensor(i) for i in range(10)]

    dataloader = DataLoader(dataset, batch_size=None, sampler=sampler)
    assert list(dataloader) == [ms.Tensor(i) for i in range(5)]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_iterdataset_single_process():
    """
    Feature: Test DataLoader with IterableDataset.
    Description: Test the DataLoader with IterableDataset in single process.
    Expectation: The result is as expected.
    """

    dataset = MyIterDataset(10)
    dataloader = DataLoader(dataset, batch_size=None)
    assert list(dataloader) == [ms.Tensor(i) for i in range(10)]


def compare_tensor_list(list1, list2):
    assert len(list1) == len(list2)
    for v1, v2 in zip(list1, list2):
        assert (v1 == v2).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mapdataset_batch():
    """
    Feature: Test DataLoader with map style dataset.
    Description: Test batch in DataLoader with map style dataset.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    dataloader = DataLoader(dataset, batch_size=1)
    compare_tensor_list(list(dataloader), [ms.Tensor([i]) for i in range(10)])

    dataloader = DataLoader(dataset, batch_size=4, drop_last=False)
    result = list(dataloader)
    expect = [ms.Tensor([0, 1, 2, 3]), ms.Tensor([4, 5, 6, 7]), ms.Tensor([8, 9])]
    compare_tensor_list(result, expect)

    dataloader = DataLoader(dataset, batch_size=4, drop_last=True)
    result = list(dataloader)
    expect = expect[:2]
    compare_tensor_list(result, expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_iterdataset_batch():
    """
    Feature: Test DataLoader with IterableDataset.
    Description: Test batch in DataLoader with IterableDataset.
    Expectation: The result is as expected.
    """

    dataset = MyIterDataset(10)
    dataloader = DataLoader(dataset, batch_size=1)
    compare_tensor_list(list(dataloader), [ms.Tensor([i]) for i in range(10)])

    dataloader = DataLoader(dataset, batch_size=4, drop_last=False)
    result = list(dataloader)
    expect = [ms.Tensor([0, 1, 2, 3]), ms.Tensor([4, 5, 6, 7]), ms.Tensor([8, 9])]
    compare_tensor_list(result, expect)

    dataloader = DataLoader(dataset, batch_size=4, drop_last=True)
    result = list(dataloader)
    expect = expect[:2]
    compare_tensor_list(result, expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_mapdataset_batch_shuffle():
    """
    Feature: Test DataLoader with map style dataset.
    Description: Test batch and shuffle in DataLoader with map style dataset.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)
    ms.set_seed(0)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, drop_last=False)
    compare_tensor_list([t.asnumpy() for t in list(dataloader)], [[0, 2, 1], [5, 9, 8], [4, 7, 6], [3]])

    ms.set_seed(1)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=True, drop_last=True)
    compare_tensor_list([t.asnumpy() for t in list(dataloader)], [[9, 0, 2], [5, 7, 4], [6, 3, 1]])


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_iterdataset_batch_shuffle():
    """
    Feature: Test DataLoader with IterableDataset.
    Description: Test batch and shuffle in DataLoader with IterableDataset.
    Expectation: The result is as expected.
    """

    dataset = MyIterDataset(10)

    with pytest.raises(ValueError) as raise_info:
        dataloader = DataLoader(dataset, batch_size=3, shuffle=True, drop_last=False)
        for data in dataloader:
            print(data, type(data))
    assert "expected unspecified shuffle option, but got shuffle=True" in str(raise_info.value)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_mapdataset_multi_process():
    """
    Feature: Test DataLoader with MapDataset.
    Description: Test the DataLoader with MapDataset in multi process.
    Expectation: The result is as expected.
    """

    dataset = MyDataset(10)

    dataloader = DataLoader(dataset, batch_size=3, num_workers=4, prefetch_factor=1)
    for data in dataloader:
        print(data)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        sampler=RandomSampler(dataset, replacement=True),
    )
    for data in dataloader:
        print(data)

    dataloader = DataLoader(dataset, batch_size=2, num_workers=12, shuffle=False)
    for data in dataloader:
        print(data)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_mapdataset_multi_process_exception():
    """
    Feature: Test DataLoader with MapDataset.
    Description: Test the DataLoader with MapDataset in multi process with exception.
    Expectation: Raise RuntimeError.
    """

    class ExceptionDataset(Dataset):
        """
        A map style dataset that raises an exception.
        """

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

    dataloader = DataLoader(dataset, batch_size=3, num_workers=4, prefetch_factor=1)
    with pytest.raises(RuntimeError, match="I got an exception!!!"):
        for _ in dataloader:
            pass


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_iterdataset_multi_process():
    """
    Feature: Test DataLoader with IterableDataset.
    Description: Test the DataLoader with IterableDataset in multi process.
    Expectation: The result is as expected.
    """

    dataset = MyIterDataset(3)

    dataloader = DataLoader(dataset, batch_size=None, num_workers=2, prefetch_factor=2)
    for data in dataloader:
        print(data)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensordataset():
    """
    Feature: Test TensorDataset.
    Description: Test iterate the TensorDataset.
    Expectation: The result is as expected.
    """

    dataset = TensorDataset(ms.Tensor([1, 2, 3, 4, 5]))
    print(len(dataset))
    for data in dataset:
        print(data)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_dataloader_iterdataset_multi_process_with_start_method():
    """
    Feature: Test DataLoader with IterableDataset.
    Description: Test the DataLoader with IterableDataset in multi process with start method.
    Expectation: The result is as expected.
    """

    dataset = MyIterDataset(3)

    dataloader = DataLoader(dataset, batch_size=None, num_workers=2, prefetch_factor=2, multiprocessing_context="fork")
    for data in dataloader:
        print(data)

    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=2,
        prefetch_factor=2,
        multiprocessing_context=multiprocessing.get_context("spawn"),
    )
    for data in dataloader:
        print(data)


def collate_fn_redundant_param(batch, redundant_param):
    """
    A collate function that takes redundant parameters.
    """
    return default_collate(batch), redundant_param


def collate_fn_missing_param():
    """
    A collate function that takes missing parameters.
    """
    return ms.Tensor([0])


def worker_init_fn_redundant_param(worker_id, redundant_param):
    """
    A worker init function that takes redundant parameters.
    """
    return worker_id, redundant_param


def worker_init_fn_missing_param():
    """
    A worker init function that takes missing parameters.
    """
    return None


class TestDataLoaderParamValidation:
    """
    Test DataLoader parameter validation.
    """

    class NotInheritIterableDataset:
        """
        A class that does not inherit from IterableDataset.
        """

        def __iter__(self):
            return iter(range(10))

    class NotImplementGetitemDataset:
        """
        A class that does not implement __getitem__.
        """

        def __len__(self):
            return 10

    @staticmethod
    def run_data_loader(**kwargs):
        """
        Run the DataLoader with the given arguments.
        """

        if "dataset" not in kwargs:
            kwargs["dataset"] = MyDataset(10)
        loader = DataLoader(**kwargs)
        for _ in loader:
            pass

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("dataset", (NotInheritIterableDataset(), NotImplementGetitemDataset()))
    def test_dataloader_invalid_dataset(self, dataset):
        """
        Feature: Test DataLoader with invalid dataset.
        Description: Test the error message when the dataset does not inherit from Dataset or IterableDataset.
        Expectation: Raise NotImplementedError.
        """

        with pytest.raises(NotImplementedError, match=" should implement `__getitem__` method if it is map style"):
            self.run_data_loader(dataset=dataset)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize(
        "batch_size_case",
        (
            (0.5, TypeError, "batch_size must be int"),
            (False, TypeError, "batch_size must be int"),
            (0, ValueError, "batch_size must be a positive integer value"),
        ),
    )
    def test_dataloader_invalid_batch_size(self, batch_size_case):
        """
        Feature: Test DataLoader with invalid batch size.
        Description: Test the error message when the batch size is not an integer.
        Expectation: Raise ValueError.
        """

        batch_size, exception, msg = batch_size_case
        with pytest.raises(exception, match=msg):
            self.run_data_loader(batch_size=batch_size)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("shuffle", (0.5, []))
    def test_dataloader_invalid_shuffle(self, shuffle):
        """
        Feature: Test DataLoader with invalid shuffle.
        Description: Test the error message when the shuffle is not a boolean.
        Expectation: Raise TypeError.
        """

        with pytest.raises(TypeError, match="shuffle must be bool"):
            self.run_data_loader(shuffle=shuffle)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize(
        "sampler_case",
        (
            (0.5, TypeError, "is not iterable"),
            ([[0], [1], [2]], TypeError, "list indices must be integers or slices, not list"),
        ),
    )
    def test_dataloader_invalid_sampler(self, sampler_case):
        """
        Feature: Test DataLoader with invalid sampler.
        Description: Test the error message when the sampler is not iterable or Iterator[int].
        Expectation: Raise TypeError.
        """

        sampler, exception, msg = sampler_case
        with pytest.raises(exception, match=msg):
            self.run_data_loader(sampler=sampler)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize(
        "batch_sampler_case",
        (
            (-1.0, TypeError, "is not iterable"),
            ([0, 1, 2], TypeError, "is not iterable"),
        ),
    )
    def test_dataloader_invalid_batch_sampler(self, batch_sampler_case):
        """
        Feature: Test DataLoader with invalid batch sampler.
        Description: Test the error message when the batch sampler is not iterable or Iterator[List[int]].
        Expectation: Raise TypeError.
        """

        batch_sampler, exception, msg = batch_sampler_case
        with pytest.raises(exception, match=msg):
            self.run_data_loader(batch_sampler=batch_sampler)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize(
        "num_workers_case",
        (
            (-1, ValueError, "must be non-negative"),
            (0.5, TypeError, "num_workers must be int"),
            (True, TypeError, "num_workers must be int"),
        ),
    )
    def test_dataloader_invalid_num_workers(self, num_workers_case):
        """
        Feature: Test DataLoader with invalid num_workers.
        Description: Test the error message when the num_workers is not an integer or non-negative.
        Expectation: Raise ValueError or TypeError.
        """

        num_workers, exception, msg = num_workers_case
        with pytest.raises(exception, match=msg):
            self.run_data_loader(num_workers=num_workers)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize(
        "collate_fn_case",
        (
            (0.5, TypeError, "is not callable"),
            (collate_fn_redundant_param, TypeError, r"missing .* required positional argument"),
            (collate_fn_missing_param, TypeError, "takes .* positional arguments but .* was given"),
        ),
    )
    def test_dataloader_invalid_collate_fn(self, collate_fn_case):
        """
        Feature: Test DataLoader with invalid collate_fn.
        Description: Test the error message when the collate_fn is not callable or arguments not match.
        Expectation: Raise TypeError.
        """

        collate_fn, exception, msg = collate_fn_case
        with pytest.raises(exception, match=msg):
            self.run_data_loader(collate_fn=collate_fn)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("pin_memory", (-1.0, 10, (0,)))
    def test_dataloader_invalid_pin_memory(self, pin_memory):
        """
        Feature: Test DataLoader with invalid pin_memory.
        Description: Test the error message when the pin_memory is not a boolean.
        Expectation: Raise TypeError.
        """

        with pytest.raises(TypeError, match="pin_memory must be bool"):
            self.run_data_loader(pin_memory=pin_memory)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("drop_last", (0.5, 0, []))
    def test_dataloader_invalid_drop_last(self, drop_last):
        """
        Feature: Test DataLoader with invalid drop_last.
        Description: Test the error message when the drop_last is not a boolean.
        Expectation: Raise TypeError.
        """

        with pytest.raises(TypeError, match="drop_last must be bool"):
            self.run_data_loader(drop_last=drop_last)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize(
        "timeout_case",
        (
            ([], TypeError, "timeout must be int or float"),
            (False, TypeError, "timeout must be int or float"),
            (-3, ValueError, "timeout must be non-negative"),
        ),
    )
    def test_dataloader_invalid_timeout(self, timeout_case):
        """
        Feature: Test DataLoader with invalid timeout.
        Description: Test the error message when the timeout is not an integer or non-negative.
        Expectation: Raise ValueError or TypeError.
        """

        timeout, exception, msg = timeout_case
        with pytest.raises(exception, match=msg):
            self.run_data_loader(timeout=timeout)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize(
        "worker_init_fn_case",
        (
            (0.5, TypeError, "is not callable"),
            (worker_init_fn_redundant_param, TypeError, r"missing .* required positional argument"),
            (
                worker_init_fn_missing_param,
                TypeError,
                "takes .* positional arguments but .* was given",
            ),
        ),
    )
    def test_dataloader_invalid_worker_init_fn(self, worker_init_fn_case):
        """
        Feature: Test DataLoader with invalid worker_init_fn.
        Description: Test the error message when the worker_init_fn is not callable or arguments not match.
        Expectation: Raise TypeError.
        """

        worker_init_fn, exception, msg = worker_init_fn_case
        with pytest.raises(exception, match=msg):
            self.run_data_loader(collate_fn=worker_init_fn)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize(
        "multiprocessing_context_context",
        (
            (
                1,
                TypeError,
                "multiprocessing_context must be a valid start method or multiprocessing.context.BaseContext",
            ),
            ("context", ValueError, r"cannot find context for .*"),
            (
                multiprocessing,
                TypeError,
                "must be a valid start method or multiprocessing.context.BaseContext",
            ),
        ),
    )
    def test_dataloader_invalid_multiprocessing_context(self, multiprocessing_context_context):
        """
        Feature: Test DataLoader with invalid multiprocessing_context.
        Description: Test the error message when the multiprocessing_context is not a valid start method or
            multiprocessing.context.BaseContext.
        Expectation: Raise ValueError or TypeError.
        """

        multiprocessing_context, exception, msg = multiprocessing_context_context
        with pytest.raises(exception, match=msg):
            self.run_data_loader(multiprocessing_context=multiprocessing_context, num_workers=1)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_dataloader_multiprocessing_context_with_invalid_num_workers(self):
        """
        Feature: Test DataLoader with invalid num_workers.
        Description: Test the error message when the num_workers is 0.
        Expectation: Raise ValueError.
        """

        with pytest.raises(ValueError,
                           match="multiprocessing_context can only be specified when num_workers is greater than 0"):
            self.run_data_loader(multiprocessing_context="fork", num_workers=0)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("multiprocessing_context",
                             ("fork", multiprocessing.get_context("fork"), ms.multiprocessing.get_context("fork")))
    def test_dataloader_warns_with_multiprocessing_fork(self, multiprocessing_context):
        """
        Feature: Test DataLoader with multiprocessing_context.
        Description: Test the warning message when the multiprocessing_context is "fork".
        Expectation: No error.
        """

        self.run_data_loader(multiprocessing_context=multiprocessing_context, num_workers=1)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("generator", (0.5, False))
    def test_dataloader_invalid_generator(self, generator):
        """
        Feature: Test DataLoader with invalid generator.
        Description: Test the error message when the generator is not a mindspore.Generator.
        Expectation: Raise TypeError.
        """

        with pytest.raises(TypeError, match="generator must be mindspore.Generator"):
            self.run_data_loader(generator=generator)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize(
        "prefetch_factor_case",
        (
            (0.5, TypeError, "prefetch_factor must be int"),
            (False, TypeError, "prefetch_factor must be int"),
            (0, ValueError, "prefetch_factor must be positive"),
        ),
    )
    def test_dataloader_invalid_prefetch_factor(self, prefetch_factor_case):
        """
        Feature: Test DataLoader with invalid prefetch_factor.
        Description: Test the error message when the prefetch_factor is not an integer or non-negative.
        Expectation: Raise ValueError or TypeError.
        """

        prefetch_factor, exception, msg = prefetch_factor_case
        with pytest.raises(exception, match=msg):
            self.run_data_loader(prefetch_factor=prefetch_factor, num_workers=1)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("persistent_workers", (0.3, -1, []))
    def test_dataloader_invalid_persistent_workers(self, persistent_workers):
        """
        Feature: Test DataLoader with invalid persistent_workers.
        Description: Test the error message when the persistent_workers is not a boolean.
        Expectation: Raise TypeError.
        """

        with pytest.raises(TypeError, match="persistent_workers must be bool"):
            self.run_data_loader(persistent_workers=persistent_workers)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_dataloader_persistent_workers_without_multiprocessing(self):
        """
        Feature: Test DataLoader with persistent_workers without multiprocessing.
        Description: Test the error message when the persistent_workers is True and num_workers is 0.
        Expectation: Raise ValueError.
        """

        with pytest.raises(
                ValueError,
                match="persistent_workers can only be specified when num_workers is greater than 0",
        ):
            self.run_data_loader(persistent_workers=True)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("pin_memory_device", ("cuda", "npu"))
    def test_dataloader_invalid_pin_memory_device(self, pin_memory_device):
        """
        Feature: Test DataLoader with invalid pin_memory_device.
        Description: Test the error message when the pin_memory_device is not a string.
        Expectation: Raise TypeError.
        """

        with pytest.raises(TypeError, match="got an unexpected keyword argument 'pin_memory_device'"):
            self.run_data_loader(pin_memory_device=pin_memory_device, pin_memory=True)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("in_order", (-10, 3.5, ()))
    def test_dataloader_invalid_in_order(self, in_order):
        """
        Feature: Test DataLoader with invalid in_order.
        Description: Test the error message when the in_order is not a boolean.
        Expectation: Raise TypeError.
        """

        with pytest.raises(TypeError, match="in_order must be bool"):
            self.run_data_loader(in_order=in_order)


class TestDataLoader:
    """
    Test DataLoader.
    """

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("drop_last", (False, True))
    @pytest.mark.parametrize("batch_size", (1, 5))
    def test_map_style_dataloader_len_with_batch_size(self, batch_size, drop_last):
        """
        Feature: Test DataLoader with batch size.
        Description: Test the length of DataLoader with batch size.
        Expectation: The length of DataLoader is the expected length.
        """

        dataset_sizes = [batch_size - 1, batch_size, batch_size + 1]
        for dataset_size in dataset_sizes:
            data_loader = DataLoader(MyDataset(dataset_size), batch_size=batch_size, drop_last=drop_last)
            if drop_last:
                expected_len = dataset_size // batch_size
            else:
                expected_len = (dataset_size - 1) // batch_size + 1
            assert len(data_loader) == expected_len
            assert sum([1 for _ in data_loader]) == expected_len

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("persistent_workers", (True, False))
    def test_iterable_style_dataloader(self, persistent_workers):
        """
        Feature: Test DataLoader with persistent_workers.
        Description: Test the result of DataLoader with persistent_workers.
        Expectation: The result of DataLoader is the expected result.
        """

        data_loader = DataLoader(MyIterDataset(3), num_workers=1, persistent_workers=persistent_workers)
        for _ in range(3):
            for index, data in enumerate(data_loader):
                assert data == ms.tensor([index])


class TestMultiprocessingDataLoader:
    """
    Test DataLoader with multiprocessing.
    """

    def setup_class(self):
        """
        Setup the DataLoader.
        """

        self.data_loader = DataLoader(MyDataset(10), num_workers=2)

    @pytest.mark.skip("Not support yet.")
    @arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("pin_memory", (False, True))
    def test_pin_memory(self, monkeypatch, pin_memory):
        """
        Feature: Test DataLoader with pin_memory.
        Description: Test the result of DataLoader with pin_memory.
        Expectation: The result of DataLoader is pinned.
        """

        monkeypatch.setattr(self.data_loader, "pin_memory", pin_memory)
        for data in self.data_loader:
            assert data.is_pinned() == pin_memory

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("persistent_workers", (False, True))
    def test_persistent_workers(self, monkeypatch, persistent_workers):
        """
        Feature: Test DataLoader with persistent_workers.
        Description: Test the result of DataLoader with persistent_workers.
        Expectation: The the worker process is the same.
        """

        monkeypatch.setattr(MyDataset, "__getitem__", lambda *inputs: os.getpid())
        monkeypatch.setattr(
            self,
            "data_loader",
            DataLoader(MyDataset(1), num_workers=2, persistent_workers=persistent_workers),
        )
        worker_ids = []
        for _ in range(2):
            for data in self.data_loader:
                worker_ids.append(data)
        assert (worker_ids[0] == worker_ids[1]) == persistent_workers

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_multiprocessing_data_loader(self):
        """
        Feature: Test DataLoader with multiprocessing.
        Description: Test the result of DataLoader with multiprocessing.
        Expectation: The result of DataLoader is the expected result.
        """

        for _ in range(3):
            result = []
            for data in self.data_loader:
                result.append(data)
            assert result == [ms.tensor([i], dtype=ms.uint8) for i in range(10)]

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_worker_raise_exception(self, monkeypatch):
        """
        Feature: Test DataLoader with worker raise exception.
        Description: Test the error message when the worker raise exception.
        Expectation: Raise RuntimeError.
        """

        def mock_getitem(self, index):
            if get_worker_info().id == 1:
                raise RuntimeError("Worker 1 raises RuntimeError!")
            return np.array(self.data[index], dtype=np.uint8)

        monkeypatch.setattr(MyDataset, "__getitem__", mock_getitem)
        with pytest.raises(RuntimeError, match="Worker 1 raises RuntimeError!"):
            for _ in self.data_loader:
                pass

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_timeout(self, monkeypatch):
        """
        Feature: Test DataLoader with timeout.
        Description: Test the error message when the timeout is reached.
        Expectation: Raise RuntimeError.
        """

        def mock_getitem(self, index):
            time.sleep(5)
            return np.array(self.data[index], dtype=np.uint8)

        monkeypatch.setattr(MyDataset, "__getitem__", mock_getitem)
        monkeypatch.setattr(self.data_loader, "timeout", 3)
        with pytest.raises(RuntimeError, match="DataLoader get data timeout after 3 seconds"):
            for _ in self.data_loader:
                pass

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("in_order", (False, True))
    def test_in_order(self, monkeypatch, in_order):
        """
        Feature: Test DataLoader with in_order.
        Description: Test the result of DataLoader with in_order.
        Expectation: The result is in order or not as in_order.
        """

        def mock_getitem(self, index):
            time.sleep(random.random())
            return np.array(self.data[index], dtype=np.uint8)

        monkeypatch.setattr(MyDataset, "__getitem__", mock_getitem)
        monkeypatch.setattr(self.data_loader, "num_workers", 4)
        monkeypatch.setattr(self.data_loader, "in_order", in_order)
        result = []
        for data in self.data_loader:
            result.append(data)
        assert (result == [ms.tensor([i], dtype=ms.uint8) for i in range(10)]) == in_order

    @pytest.mark.skip("Not support yet.")
    @arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    def test_kill_pin_memory_thread(self, monkeypatch):
        """
        Feature: Test DataLoader with kill pin memory thread.
        Description: Test the error message when the pin memory thread is killed.
        Expectation: Raise RuntimeError.
        """

        monkeypatch.setattr(self.data_loader, "pin_memory", True)

        def mock_getitem(self, index):
            time.sleep(5)
            return np.array(self.data[index], dtype=np.uint8)

        monkeypatch.setattr(MyDataset, "__getitem__", mock_getitem)

        with pytest.raises(RuntimeError, match="Pin memory thread exited unexpectedly"):
            for _ in self.data_loader:
                self.data_loader._iterator._pin_memory_thread._stop()  # pylint: disable=protected-access

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("sig", (signal.SIGKILL, signal.SIGTERM, signal.SIGINT))
    def test_kill_worker_process(self, monkeypatch, sig):
        """
        Feature: Test DataLoader with kill worker process.
        Description: Test the error message when the worker process is killed.
        Expectation: Raise RuntimeError.
        """

        def mock_getitem(self, index):
            time.sleep(1)
            return np.array(self.data[index], dtype=np.uint8)

        monkeypatch.setattr(MyDataset, "__getitem__", mock_getitem)
        monkeypatch.setattr(self.data_loader, "num_workers", 4)

        multiprocess_data_loader = iter(self.data_loader)
        worker_group = multiprocess_data_loader.data_workers
        assert len(worker_group) == 4
        with pytest.raises(RuntimeError, match=r"DataLoader worker .* exited unexpectedly"):
            for _ in multiprocess_data_loader:
                os.kill(worker_group[0].pid, sig)

    @arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
    @pytest.mark.parametrize("sig", (signal.SIGKILL, signal.SIGINT))
    def test_kill_main_process(self, sig):
        """
        Feature: Test DataLoader with kill main process.
        Description: Test the error message when the main process is killed.
        Expectation: Raise RuntimeError.
        """

        script_path = os.path.join(os.path.dirname(__file__), "run_data_loader.py")
        process = subprocess.Popen(["python", script_path])
        time.sleep(3)
        assert psutil.pid_exists(process.pid)

        child_processes = []
        try:
            parent = psutil.Process(process.pid)
            child_processes = []
            while len(child_processes) != 8:
                child_processes = [child.pid for child in parent.children(recursive=True)]

            os.kill(process.pid, sig)
            process.wait(timeout=2)
            assert not psutil.pid_exists(process.pid)

            start_time = time.time()
            while time.time() - start_time < 5:
                remaining = [pid for pid in child_processes if psutil.pid_exists(pid)]
                if not remaining:
                    break
                time.sleep(0.1)
            else:
                alive = [pid for pid in child_processes if psutil.pid_exists(pid)]
                pytest.fail(f"Worker process is not finished: {alive}")
        finally:
            if psutil.pid_exists(process.pid):
                os.kill(process.pid, signal.SIGKILL)
            for worker_pid in child_processes:
                if psutil.pid_exists(worker_pid):
                    os.kill(worker_pid, signal.SIGKILL)
