# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
import copy
import os
import random
import re
import subprocess
import time

import psutil
import pytest
import numpy as np

import mindspore
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.engine.iterators as it
import mindspore.ops as ops
from mindspore import log as logger
from mindspore import Tensor
from util import config_get_set_seed, save_and_check_dict


# Generate 1d int numpy array from 0 - 63
def generator_1d():
    for i in range(64):
        yield (np.array([i]),)


class DatasetGeneratorTwoLevelPipeline:
    def __init__(self):
        self.data = np.array([1, 2, 3, 4, 5, 6])
        self.dataset = ds.GeneratorDataset(self.data, column_names=["col1"], shuffle=False)

    def __getitem__(self, i):
        return self.dataset[i]

    def __len__(self):
        return self.dataset.get_dataset_size()


class DatasetGenerator:
    def __init__(self):
        pass

    def __getitem__(self, item):
        return (np.array([item]),)

    def __len__(self):
        return 10


class DatasetGeneratorLarge:
    def __init__(self):
        self.data = np.array(range(4000))

    def __getitem__(self, item):
        return (self.data + item, self.data * 10)

    def __len__(self):
        return 10


class DatasetGeneratorSmall:
    def __init__(self):
        self.data = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


class DatasetGeneratorMixed:
    def __init__(self):
        pass

    def __getitem__(self, item):
        flatten = ops.Flatten()
        x = Tensor(np.ones(shape=[2, 3]), mindspore.float32)
        output = flatten(x)
        return (output.asnumpy(),)

    def __len__(self):
        return 10


class CustomizedData:
    def __init__(self, low, mid, high):
        self.input_ids = np.ones((low, mid, high), dtype=np.int32)

    def __getitem__(self, index):
        return self.input_ids

    def __len__(self):
        return 10


def test_generator_0():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator : 0 - 63")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1


# Generate md int numpy array from [[0, 1], [2, 3]] to [[63, 64], [65, 66]]
def generator_md():
    for i in range(64):
        yield (np.array([[i, i + 1], [i + 2, i + 3]]),)


def test_generator_1():
    """
    Feature: GeneratorDataset
    Description: Test MD Generator with shape [2, 2]
    Expectation: The dataset is processed as expected
    """
    logger.info("Test MD Generator : 0 - 63, with shape [2, 2]")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_md, ["data"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1


# Generate two columns, the first column is from Generator1D, the second column is from GeneratorMD
def generator_mc(maxid=64):
    for i in range(maxid):
        yield (np.array([i]), np.array([[i, i + 1], [i + 2, i + 3]]))


def test_generator_2():
    """
    Feature: GeneratorDataset
    Description: Test multi column Generator
    Expectation: The dataset is processed as expected
    """
    logger.info("Test multi column generator")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc, ["col0", "col1"])

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["col0"], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["col1"], golden)
        i = i + 1


def test_generator_3():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator with repeat(4)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator : 0 - 63 + Repeat(4)")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    data1 = data1.repeat(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1
        if i == 64:
            i = 0


def test_generator_4():
    """
    Feature: GeneratorDataset
    Description: Test fixed size 1D Generator with batch(4)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator : 0 - 63 + batch(4)")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_1d, ["data"])

    data1 = data1.batch(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 4


def generator_with_type(t):
    for i in range(64):
        yield (np.array([i], dtype=t),)


def type_tester(t):
    logger.info("Test with Type {}".format(t.__name__))

    # apply dataset operations
    data1 = ds.GeneratorDataset((lambda: generator_with_type(t)), ["data"])

    data1 = data1.batch(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]], dtype=t)
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 4


def test_generator_5():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator on different data types
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator on all data types")

    types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32, np.float64]

    for t in types:
        type_tester(t)


def type_tester_with_type_check(t, c):
    logger.info("Test with Type {}".format(t.__name__))

    # apply dataset operations
    data1 = ds.GeneratorDataset((lambda: generator_with_type(t)), ["data"], column_types=[c])

    data1 = data1.batch(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]], dtype=t)
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 4


def test_generator_6():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator on different data types with type check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator on all data types with type check")

    np_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32,
                np.float64]
    de_types = [mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.uint8, mstype.uint16, mstype.uint32,
                mstype.uint64, mstype.float32, mstype.float64]

    for i, _ in enumerate(np_types):
        type_tester_with_type_check(np_types[i], de_types[i])


def generator_with_type_2c(t):
    for i in range(64):
        yield (np.array([i], dtype=t), np.array([i], dtype=t))


def type_tester_with_type_check_2c(t, c):
    logger.info("Test with Type {}".format(t.__name__))

    # apply dataset operations
    data1 = ds.GeneratorDataset((lambda: generator_with_type_2c(t)), ["data0", "data1"], column_types=c)

    data1 = data1.batch(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]], dtype=t)
        np.testing.assert_array_equal(item["data0"], golden)
        i = i + 4


def test_generator_7():
    """
    Feature: GeneratorDataset
    Description: Test 2 column Generator on different data type with type check
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 2 column Generator on all data types with type check")

    np_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32,
                np.float64]
    de_types = [mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.uint8, mstype.uint16, mstype.uint32,
                mstype.uint64, mstype.float32, mstype.float64]

    for i, _ in enumerate(np_types):
        type_tester_with_type_check_2c(np_types[i], [None, de_types[i]])


def test_generator_8():
    """
    Feature: GeneratorDataset
    Description: Test multi column Generator with few mapops
    Expectation: The dataset is processed as expected
    """
    logger.info("Test multi column generator with mapops to check the order too")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: x * 3), input_columns="col0", output_columns="out0",
                      num_parallel_workers=2)
    data1 = data1.map(operations=(lambda x: (x * 7, x)), input_columns="col1", output_columns=["out1", "out2"],
                      num_parallel_workers=2)
    data1 = data1.project(["out0", "out1", "out2"])
    data1 = data1.map(operations=(lambda x: x + 1), input_columns="out2", output_columns="out2",
                      num_parallel_workers=2)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i * 3])
        np.testing.assert_array_equal(item["out0"], golden)
        golden = np.array([[i * 7, (i + 1) * 7], [(i + 2) * 7, (i + 3) * 7]])
        np.testing.assert_array_equal(item["out1"], golden)
        golden = np.array([[i + 1, i + 2], [i + 3, i + 4]])
        np.testing.assert_array_equal(item["out2"], golden)
        i = i + 1


def test_generator_9():
    """
    Feature: GeneratorDataset
    Description: Test map column order when len(input_columns) == len(output_columns)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when len(input_columns) == len(output_columns).")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["image", "label"])
    data2 = ds.GeneratorDataset(generator_mc(2048), ["label", "image"])
    data1 = data1.map(operations=(lambda x: x * 3), input_columns="label",
                      num_parallel_workers=4)
    data2 = data2.map(operations=(lambda x: x * 3), input_columns="label",
                      num_parallel_workers=4)

    # Expected column order is not changed.
    i = 0
    for data1, data2 in zip(data1, data2):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data1[0].asnumpy(), golden)
        golden = np.array([[i * 3, (i + 1) * 3], [(i + 2) * 3, (i + 3) * 3]])
        np.testing.assert_array_equal(data1[1].asnumpy(), golden)

        golden = np.array([i * 3])
        np.testing.assert_array_equal(data2[0].asnumpy(), golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(data2[1].asnumpy(), golden)
        i = i + 1


def test_generator_10():
    """
    Feature: GeneratorDataset
    Description: Test map column order when len(input_columns) != len(output_columns)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when len(input_columns) != len(output_columns).")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: (x, x * 5)), input_columns="col1", output_columns=["out1", "out2"],
                      num_parallel_workers=2)
    data1 = data1.project(['col0', 'out1', 'out2'])

    # Expected column order is |col0|out1|out2|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        golden = np.array([i])
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[1], golden)
        golden = np.array([[i * 5, (i + 1) * 5], [(i + 2) * 5, (i + 3) * 5]])
        np.testing.assert_array_equal(item[2], golden)
        i = i + 1


def test_generator_11():
    """
    Feature: GeneratorDataset
    Description: Test .project drops some columns
    Expectation: The dataset is processed as expected
    """
    logger.info("Test .project drops some columns.")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: (x, x * 5)), input_columns="col1", output_columns=["out1", "out2"],
                      num_parallel_workers=2)
    data1 = data1.project(["out1", "out2"])

    # Expected column order is |out1|out2|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        # len should be 2 because col0 is dropped
        assert len(item) == 2
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array([[i * 5, (i + 1) * 5], [(i + 2) * 5, (i + 3) * 5]])
        np.testing.assert_array_equal(item[1], golden)
        i = i + 1


def test_generator_12():
    """
    Feature: GeneratorDataset
    Description: Test map column order when input_columns and output_columns are None
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when input_columns and output_columns are None.")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: (x * 5)), num_parallel_workers=2)

    # Expected column order is |col0|col1|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[1], golden)
        i = i + 1

    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: (x * 5)), num_parallel_workers=2)
    data1 = data1.project(["col1", "col0"])

    # Expected column order is |col0|col1|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item[1], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[0], golden)
        i = i + 1


def test_generator_13():
    """
    Feature: GeneratorDataset
    Description: Test map column order when input_columns is None
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when input_columns is None.")

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"])
    data1 = data1.map(operations=(lambda x: (x * 5)), output_columns=["out0"], num_parallel_workers=2)

    # Expected column order is |out0|col1|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[1], golden)
        i = i + 1

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # len should be 2 because col0 is dropped
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item["out0"], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item["col1"], golden)
        i = i + 1


def test_generator_14():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator MP with CPP sampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator MP : 0 - 63")
    # Sometimes there are some ITERATORS left in ITERATORS_LIST when run all UTs together,
    # and cause core dump and blocking in this UT. Add cleanup() here to fix it.
    it._cleanup()  # pylint: disable=W0212

    # Reduce memory needed by reducing queue size
    prefetch_original = ds.config.get_prefetch_size()
    ds.config.set_prefetch_size(1)

    source = [(np.array([x]),) for x in range(256)]
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=ds.SequentialSampler(),
                              num_parallel_workers=4, max_rowsize=1).repeat(2)
    i = 0
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data["data"], golden)
        i = i + 1
        if i == 256:
            i = 0

    ds.config.set_prefetch_size(prefetch_original)


def test_generator_15():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator MP with Python sampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator MP : 0 - 63")

    ## Reduce memory needed by reducing queue size
    prefetch_original = ds.config.get_prefetch_size()
    ds.config.set_prefetch_size(1)

    sampler = [x for x in range(256)]
    source = [(np.array([x]),) for x in range(256)]
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=sampler,
                              num_parallel_workers=4, max_rowsize=1).repeat(1)
    i = 0
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data["data"], golden)
        i = i + 1
        if i == 256:
            i = 0

    ds.config.set_prefetch_size(prefetch_original)


def test_generator_16():
    """
    Feature: GeneratorDataset
    Description: Test multi column generator Mp with CPP sampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test multi column generator")

    source = [(np.array([x]), np.array([x + 1])) for x in range(256)]
    # apply dataset operations
    data1 = ds.GeneratorDataset(source, ["col0", "col1"], sampler=ds.SequentialSampler())

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["col0"], golden)
        golden = np.array([i + 1])
        np.testing.assert_array_equal(item["col1"], golden)
        i = i + 1


def test_generator_17():
    """
    Feature: GeneratorDataset
    Description: Test multi column generator Mp with CPP sampler
    Expectation: The dataset is processed as expected
    """
    logger.info("Test multi column generator")

    sampler = [x for x in range(256)]
    source = [(np.array([x]), np.array([x + 1])) for x in range(256)]
    # apply dataset operations
    data1 = ds.GeneratorDataset(source, ["col0", "col1"], sampler=sampler)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(item["col0"], golden)
        golden = np.array([i + 1])
        np.testing.assert_array_equal(item["col1"], golden)
        i = i + 1


def test_generator_18():
    """
    Feature: GeneratorDataset
    Description: Test multiprocessing flag (same as test 13 with python_multiprocessing=True flag)
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map column order when input_columns is None.")

    # Reduce shm usage by disabling this optimization
    mem_original = ds.config.get_enable_shared_mem()
    ds.config.set_enable_shared_mem(False)

    # apply dataset operations
    data1 = ds.GeneratorDataset(generator_mc(2048), ["col0", "col1"], python_multiprocessing=True)
    data1 = data1.map(operations=(lambda x: (x * 5)), output_columns=["out0"], num_parallel_workers=2,
                      python_multiprocessing=True)

    # Expected column order is |out0|col1|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array([[i, i + 1], [i + 2, i + 3]])
        np.testing.assert_array_equal(item[1], golden)
        i = i + 1

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        # len should be 2 because col0 is dropped
        assert len(item) == 2
        golden = np.array([i * 5])
        np.testing.assert_array_equal(item["out0"], golden)
        i += 1

    ds.config.set_enable_shared_mem(mem_original)


def test_generator_19():
    """
    Feature: GeneratorDataset
    Description: Test multiprocessing 2 different large columns
    Expectation: The dataset is processed as expected
    """
    logger.info("Test map multiprocessing 2 different large columns.")

    # apply dataset operations
    data1 = ds.GeneratorDataset(DatasetGeneratorLarge(), ["col0", "col1"], python_multiprocessing=True, shuffle=False)

    # Expected column order is |out0|col1|
    i = 0
    for item in data1.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert len(item) == 2
        golden = np.array(range(4000)) + i
        np.testing.assert_array_equal(item[0], golden)
        golden = np.array(range(4000)) * 10
        np.testing.assert_array_equal(item[1], golden)
        i = i + 1


class RandomAccessDataset:
    def __init__(self):
        self.__data = np.random.sample((5, 1))

    def __getitem__(self, item):
        return self.__data[item]

    def __len__(self):
        return 5


class RandomAccessDatasetWithoutLen:
    def __init__(self):
        self.__data = np.random.sample((5, 1))

    def __getitem__(self, item):
        return self.__data[item]


class IterableDataset:
    def __init__(self):
        self.count = 0
        self.max = 10

    def __iter__(self):
        return self

    def __next__(self):
        if self.count >= self.max:
            raise StopIteration
        self.count += 1
        return (np.array(self.count),)


def test_generator_20():
    """
    Feature: GeneratorDataset
    Description: Test mappable and unmappable dataset as source for GeneratorDataset
    Expectation: The dataset is processed as expected
    """
    logger.info("Test mappable and unmappable dataset as source for GeneratorDataset.")

    # Mappable dataset
    data1 = ds.GeneratorDataset(RandomAccessDataset(), ["col0"])
    dataset_size1 = data1.get_dataset_size()
    assert dataset_size1 == 5

    # Mappable dataset without __len__
    data2 = ds.GeneratorDataset(RandomAccessDatasetWithoutLen(), ["col0"])
    try:
        data2.get_dataset_size()
    except RuntimeError as e:
        assert "'__len__' method is required" in str(e)

    # Unmappable dataset
    data3 = ds.GeneratorDataset(IterableDataset(), ["col0"])
    dataset_size3 = data3.get_dataset_size()
    assert dataset_size3 == 10


def test_generator_error_1():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with no data type of the converted NumPy array
    Expectation: Error is raised as expected
    """

    def generator_np():
        for i in range(64):
            yield (np.array([{i}]),)

    with pytest.raises(RuntimeError) as info:
        data1 = ds.GeneratorDataset(generator_np, ["data"])
        for _ in data1:
            pass
    assert "Data type of 1th item of the input or its converted Numpy array is expected" in str(info.value)


def test_generator_error_2():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with no data type of 1th item of the input
    Expectation: Error is raised as expected
    """

    def generator_np():
        for i in range(64):
            yield ({i},)

    with pytest.raises(RuntimeError) as info:
        data1 = ds.GeneratorDataset(generator_np, ["data"])
        for _ in data1:
            pass
    assert "Data type of 1th item of the input or its converted Numpy array is expected" in str(info.value)


def test_generator_error_4():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset when number of columns in map does not match output_columns
    Expectation: Error is raised as expected
    """
    with pytest.raises(RuntimeError) as info:
        # apply dataset operations
        data1 = ds.GeneratorDataset(generator_mc(2048), ["label", "image"])
        data1 = data1.map(operations=(lambda x: (x, x * 5)), input_columns=["label"],
                          num_parallel_workers=2)

        for _ in data1:
            pass
    assert "the number of columns returned in 'map' operations should match the number of 'output_columns'" \
           in str(info.value)


def test_generator_sequential_sampler():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with SequentialSampler
    Expectation: The dataset is processed as expected
    """
    source = [(np.array([x]),) for x in range(64)]
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=ds.SequentialSampler())
    i = 0
    for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([i])
        np.testing.assert_array_equal(data["data"], golden)
        i = i + 1


def test_generator_random_sampler():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with random sampler
    Expectation: The dataset is processed as expected
    """
    source = [(np.array([x]),) for x in range(64)]
    ds1 = ds.GeneratorDataset(source, ["data"], shuffle=True)
    for _ in ds1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        pass


def test_generator_distributed_sampler():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with distributed sampler
    Expectation: The dataset is processed as expected
    """
    source = [(np.array([x]),) for x in range(64)]
    for sid in range(8):
        ds1 = ds.GeneratorDataset(source, ["data"], shuffle=False, num_shards=8, shard_id=sid)
        i = sid
        for data in ds1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.array([i])
            np.testing.assert_array_equal(data["data"], golden)
            i = i + 8


def test_generator_num_samples():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with num_samples parameter
    Expectation: The dataset is processed as expected
    """
    source = [(np.array([x]),) for x in range(64)]
    num_samples = 32
    ds1 = ds.GeneratorDataset(source, ["data"], sampler=ds.SequentialSampler(num_samples=num_samples))
    ds2 = ds.GeneratorDataset(source, ["data"], sampler=[i for i in range(32)], num_samples=num_samples)
    ds3 = ds.GeneratorDataset(generator_1d, ["data"], num_samples=num_samples)

    count = 0
    for _ in ds1.create_dict_iterator(num_epochs=1):
        count = count + 1
    assert count == num_samples

    count = 0
    for _ in ds2.create_dict_iterator(num_epochs=1):
        count = count + 1
    assert count == num_samples

    count = 0
    for _ in ds3.create_dict_iterator(num_epochs=1):
        count = count + 1
    assert count == num_samples


def test_generator_num_samples_underflow():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with underflowed num_samples parameter
    Expectation: The dataset is processed as expected
    """
    source = [(np.array([x]),) for x in range(64)]
    num_samples = 256
    ds2 = ds.GeneratorDataset(source, ["data"], sampler=[i for i in range(64)], num_samples=num_samples)
    ds3 = ds.GeneratorDataset(generator_1d, ["data"], num_samples=num_samples)

    count = 0
    for _ in ds2.create_dict_iterator(num_epochs=1):
        count = count + 1
    assert count == 64

    count = 0
    for _ in ds3.create_dict_iterator(num_epochs=1):
        count = count + 1
    assert count == 64


def type_tester_with_type_check_2c_schema(t, c):
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset with type check 2c
    Expectation: The dataset is processed as expected
    """
    logger.info("Test with Type {}".format(t.__name__))

    schema = ds.Schema()
    schema.add_column("data0", c[0])
    schema.add_column("data1", c[1])

    # apply dataset operations
    data1 = ds.GeneratorDataset((lambda: generator_with_type_2c(t)), schema=schema)

    data1 = data1.batch(4)

    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        golden = np.array([[i], [i + 1], [i + 2], [i + 3]], dtype=t)
        np.testing.assert_array_equal(item["data0"], golden)
        i = i + 4


def test_generator_schema():
    """
    Feature: GeneratorDataset
    Description: Test 2 column Generator on different data type with type check with schema input
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 2 column Generator on all data types with type check")

    np_types = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.float32,
                np.float64]
    de_types = [mstype.int8, mstype.int16, mstype.int32, mstype.int64, mstype.uint8, mstype.uint16, mstype.uint32,
                mstype.uint64, mstype.float32, mstype.float64]

    for i, _ in enumerate(np_types):
        type_tester_with_type_check_2c_schema(np_types[i], [de_types[i], de_types[i]])


def test_generator_dataset_size_0():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset get_dataset_size by iterator method
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator : 0 - 63 get_dataset_size")

    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data_size = data1.get_dataset_size()

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
        num_rows = num_rows + 1
    assert data_size == num_rows


def test_generator_dataset_size_1():
    """
    Feature: GeneratorDataset
    Description: Test GeneratorDataset get_dataset_size by __len__ method
    Expectation: The dataset is processed as expected
    """
    logger.info("Test DatasetGenerator get_dataset_size")

    dataset_generator = DatasetGenerator()
    data1 = ds.GeneratorDataset(dataset_generator, ["data"])

    data_size = data1.get_dataset_size()

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_rows = num_rows + 1
    assert data_size == num_rows


def test_generator_dataset_size_2():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator with repeat get_dataset_size
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator + repeat get_dataset_size")

    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.repeat(2)

    data_size = data1.get_dataset_size()

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_rows = num_rows + 1
    assert data_size == num_rows


def test_generator_dataset_size_3():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator with batch get_dataset_size
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator + batch get_dataset_size")

    data1 = ds.GeneratorDataset(generator_1d, ["data"])
    data1 = data1.batch(4)

    data_size = data1.get_dataset_size()

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1):
        num_rows += 1
    assert data_size == num_rows


def test_generator_dataset_size_4():
    """
    Feature: GeneratorDataset
    Description: Test 1D Generator with num_shards get_dataset_size
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator : 0 - 63 + num_shards get_dataset_size")

    dataset_generator = DatasetGenerator()
    data1 = ds.GeneratorDataset(dataset_generator, ["data"], num_shards=3, shard_id=0)
    data_size = data1.get_dataset_size()

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        num_rows = num_rows + 1
    assert data_size == num_rows


def test_generator_dataset_size_5():
    """
    Feature: GeneratorDataset
    Description: Test get_dataset_size after create_dict_iterator
    Expectation: The dataset is processed as expected
    """
    logger.info("Test get_dataset_size after create_dict_iterator")

    dataset_generator = DatasetGenerator()
    data1 = ds.GeneratorDataset(dataset_generator, ["data"], num_shards=3, shard_id=0)

    num_rows = 0
    for _ in data1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        num_rows = num_rows + 1
    data_size = data1.get_dataset_size()
    assert data_size == num_rows


def manual_test_generator_keyboard_interrupt():
    """
    Feature: GeneratorDataset
    Description: Test keyboard_interrupt
    Expectation: The dataset is processed as expected
    """
    logger.info("Test 1D Generator MP : 0 - 63")

    class MyDS():
        def __getitem__(self, item):
            while True:
                pass

        def __len__(self):
            return 1024

    ds1 = ds.GeneratorDataset(MyDS(), ["data"], num_parallel_workers=4).repeat(2)
    for _ in ds1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
        pass


def test_explicit_deepcopy():
    """
    Feature: NumPyDataset
    Description: Test explicit_deepcopy
    Expectation: The dataset is processed as expected
    """
    logger.info("Test explicit_deepcopy")

    ds1 = ds.NumpySlicesDataset([1, 2], shuffle=False)
    ds2 = copy.deepcopy(ds1)
    for d1, d2 in zip(ds1, ds2):
        assert d1 == d2


def test_func_generator_dataset_005():
    """
    Feature: GeneratorDataset
    Description: Test Generator's class __getitem__
    Expectation: The dataset is processed as expected
    """
    result = [np.random.randn(242, 242, 242), np.random.randn(42, 24, 442)]

    class MyData():
        def __init__(self, input_para):
            self.data = input_para

        def __getitem__(self, item):
            return (Tensor(self.data[0]), Tensor(self.data[1]))

        def __len__(self):
            return 2

    column_names = ["col1", "col2"]
    dataset = ds.GeneratorDataset(MyData(result), column_names)
    i = 0
    for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert "col1" in str(data.keys())
        assert (data["col1"] == result[0]).all()
        assert (data["col2"] == result[1]).all()
        i += 1
    assert i == 2


def test_func_generator_dataset_with_zip_source():
    """
    Feature: Verify the source is zip
    Description: The source input is zip
    Expectation: Success
    """

    def synthetic_data(w, b, num_examples):
        """生成 y = Xw + b + 噪声。"""
        X = np.random.normal(0, 1, (num_examples, len(w)))
        y = np.matmul(X, w) + b
        y += np.random.normal(0, 0.01, y.shape)
        return X.astype(np.float32), y.reshape((-1, 1)).astype(np.float32)

    true_w = np.array([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 10)

    def load_array(data_arrays, column_names, batch_size, is_train=True):
        """构造一个MindSpore数据迭代器。"""
        dataset = ds.GeneratorDataset(data_arrays, column_names, shuffle=is_train)
        dataset = dataset.batch(batch_size)
        return dataset

    batch_size = 2
    dataset = load_array(zip(features, labels), ['features', 'labels'], batch_size)

    count = 0
    epochs = 10
    dataset_iter = dataset.create_dict_iterator(num_epochs=epochs, output_numpy=True)
    for _ in range(epochs):
        for _ in dataset_iter:
            count += 1
    assert count == 50


def test_generator_mixed_operator():
    """
    Feature: Test adding computing operator into user defined dataset
    Description: Will decrease num_parallel_worker into 1
    Expectation: Success
    """
    logger.info("Test adding computing operator into user defined dataset.")

    if "MS_INDEPENDENT_DATASET" in os.environ and os.environ["MS_INDEPENDENT_DATASET"].lower() == "true":
        logger.info("Mixed operator with Tensor and ops is no needed to run in independent dataset mode.")
        return

    # create dataset
    data1 = ds.GeneratorDataset(DatasetGeneratorMixed(), ["col0"], shuffle=False, python_multiprocessing=False)
    assert data1.num_parallel_workers == 1

    for _ in data1.create_tuple_iterator(num_epochs=1):
        pass


def test_generator_single_input_0():
    """
    Feature: Test single int input
    Description: Input int
    Expectation: Success
    """

    def generator_int():
        for i in range(64):
            yield i

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [i for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item]

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [i for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_0(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.array(i)
            np.testing.assert_equal(item["data"], golden)
            i = i + 1

    assert_generator_single_input_0(generator_int)
    assert_generator_single_input_0(RandomAccessDatasetInner())
    assert_generator_single_input_0(SequentialAccessDataset())


def test_generator_single_input_1():
    """
    Feature: Test single float input
    Description: Input float
    Expectation: Success
    """

    def generator_float():
        for i in range(64):
            yield i * 0.1

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [i for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item] * 0.1

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [i for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index] * 0.1
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_1(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0.0
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.array(i)
            np.testing.assert_almost_equal(item["data"], golden)
            i = i + 0.1

    assert_generator_single_input_1(generator_float)
    assert_generator_single_input_1(RandomAccessDatasetInner())
    assert_generator_single_input_1(SequentialAccessDataset())


def test_generator_single_input_2():
    """
    Feature: Test single str input
    Description: Input str
    Expectation: Success
    """

    def generator_str():
        for i in range(64):
            yield chr(ord('a') + i)

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [i for i in range(64)]

        def __getitem__(self, item):
            return chr(ord('a') + self.__data[item])

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [i for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = chr(ord('a') + self.__data[self.__index])
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_2(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            s = chr(ord('a') + i)
            golden = np.array(s)
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 1

    assert_generator_single_input_2(generator_str)
    assert_generator_single_input_2(RandomAccessDatasetInner())
    assert_generator_single_input_2(SequentialAccessDataset())


def test_generator_single_input_3():
    """
    Feature: Test single bytes input
    Description: Input bytes
    Expectation: Success
    """

    def generator_bytes():
        for i in range(64):
            yield bytes('a' * i, encoding='UTF-8')

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [bytes('a' * i, encoding='UTF-8') for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item]

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [bytes('a' * i, encoding='UTF-8') for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_3(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            b = bytes('a' * i, encoding='UTF-8')
            golden = np.array(b)
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 1

    assert_generator_single_input_3(generator_bytes)
    assert_generator_single_input_3(RandomAccessDatasetInner())
    assert_generator_single_input_3(SequentialAccessDataset())


def test_generator_single_input_4():
    """
    Feature: Test single Tensor input
    Description: Input Tensor
    Expectation: Success
    """

    def generator_tensor():
        for i in range(64):
            yield Tensor(i)

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [Tensor(i) for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item]

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [Tensor(i) for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_4(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0
        for item in data1.create_dict_iterator(num_epochs=1):  # each data is a dictionary
            golden = Tensor(i)
            assert item["data"] == golden
            i = i + 1

    assert_generator_single_input_4(generator_tensor)
    assert_generator_single_input_4(RandomAccessDatasetInner())
    assert_generator_single_input_4(SequentialAccessDataset())


def test_generator_single_input_5():
    """
    Feature: Test single np.array input
    Description: Input np.array
    Expectation: Success
    """

    def generator_np():
        for i in range(64):
            yield np.ones(i)

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [np.ones(i) for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item]

        def __len__(self):
            return 64

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = [np.ones(i) for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_5(data):
        # apply dataset operations
        data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
        i = 0
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
            golden = np.ones(i)
            np.testing.assert_array_equal(item["data"], golden)
            i = i + 1

    assert_generator_single_input_5(generator_np)
    assert_generator_single_input_5(RandomAccessDatasetInner())
    assert_generator_single_input_5(SequentialAccessDataset())


def test_generator_single_input_6():
    """
    Feature: Test single np.array input whose dtype is object
    Description: Input np.array
    Expectation: Throw exception
    """

    def generator_nested_np():
        for i in range(64):
            yield np.array([[i, i + 1, None], [i, i + 1, i + 2]])

    class RandomAccessDatasetInner:
        def __init__(self):
            self.__data = [np.array([[i, i + 1, None], [i, i + 1, i + 2]]) for i in range(64)]

        def __getitem__(self, item):
            return self.__data[item]

        def __len__(self):
            return 64

    class SequentialAccessDatasetInner:
        def __init__(self):
            self.__data = [np.array([[i, i + 1, None], [i, i + 1, i + 2]]) for i in range(64)]
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    def assert_generator_single_input_6(data):
        # apply dataset operations

        with pytest.raises(RuntimeError) as info:
            data1 = ds.GeneratorDataset(data, ["data"], shuffle=False)
            for _ in data1.create_dict_iterator(num_epochs=1, output_numpy=True):  # each data is a dictionary
                pass
        assert " Data type of the input or its converted Numpy array is expected" in str(info.value)

    assert_generator_single_input_6(generator_nested_np)
    assert_generator_single_input_6(RandomAccessDatasetInner())
    assert_generator_single_input_6(SequentialAccessDatasetInner())


def test_generator_one_dimensional_numpy_input():
    """
    Feature: Test one-dimensional numpy.int32 input
    Description: The input source data is a one-dimensional numpy array of type numpy.int32
    Expectation: No error was reported, and the iteration succeeded
    """

    class SequentialAccessDataset:
        def __init__(self):
            self.__data = np.array([i for i in range(64)], dtype=np.int32)
            self.__index = 0

        def __next__(self):
            if self.__index >= 64:
                raise StopIteration
            item = self.__data[self.__index]
            self.__index += 1
            return item

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 64

    data1 = ds.GeneratorDataset(SequentialAccessDataset(), ["data"], shuffle=False)
    i = 0
    for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
        golden = np.array(i, dtype=np.int32)
        np.testing.assert_equal(item["data"], golden)
        i = i + 1


def test_generator_with_seed_5489_when_dist():
    """
    Feature: With default seed (5489) when distributed
    Description: Default seed is 5489
    Expectation: Shuffle by seed 5489 and shard
    """

    data1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    data2 = np.array([5, 6, 7, 8], dtype=np.uint8)
    data3 = np.array([9, 10, 11, 12], dtype=np.uint8)
    data4 = np.array([13, 14, 15, 16], dtype=np.uint8)
    data5 = np.array([17, 18, 19, 20], dtype=np.uint8)
    data6 = np.array([21, 22, 23, 24], dtype=np.uint8)
    data7 = np.array([25, 26, 27, 28], dtype=np.uint8)
    data8 = np.array([29, 30, 31, 32], dtype=np.uint8)
    data9 = np.array([33, 34, 35, 36], dtype=np.uint8)
    data10 = np.array([37, 38, 39, 40], dtype=np.uint8)
    data11 = np.array([41, 42, 43, 44], dtype=np.uint8)
    data12 = np.array([45, 46, 47, 48], dtype=np.uint8)
    data13 = np.array([49, 50, 51, 52], dtype=np.uint8)
    data14 = np.array([53, 54, 55, 56], dtype=np.uint8)
    data15 = np.array([57, 58, 59, 60], dtype=np.uint8)
    data16 = np.array([61, 62, 63, 64], dtype=np.uint8)
    data17 = np.array([65, 66, 67, 68], dtype=np.uint8)
    data18 = np.array([69, 70, 71, 72], dtype=np.uint8)
    data19 = np.array([73, 74, 75, 76], dtype=np.uint8)
    data20 = np.array([77, 78, 79, 80], dtype=np.uint8)

    data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10,
            data11, data12, data13, data14, data15, data16, data17, data18, data19, data20]

    label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    assert ds.config.get_seed() == 5489

    dataset = ds.NumpySlicesDataset((data, label), ["data", "label"], num_shards=4, shard_id=2)
    dataset = dataset.batch(batch_size=2)

    save_and_check_dict(dataset, "test_seed_when_distributed_01.npz", False)


def test_generator_with_set_seed_when_dist():
    """
    Feature: Test GeneratorDataset with ds.config.set_seed(4321) when distributed
    Description: Support ds.config.set_seed when user use ds.config.set_seed(4321)
    Expectation: Shuffle by seed 4321 and shard
    """

    data1 = np.array([1, 2, 3, 4], dtype=np.uint8)
    data2 = np.array([5, 6, 7, 8], dtype=np.uint8)
    data3 = np.array([9, 10, 11, 12], dtype=np.uint8)
    data4 = np.array([13, 14, 15, 16], dtype=np.uint8)
    data5 = np.array([17, 18, 19, 20], dtype=np.uint8)
    data6 = np.array([21, 22, 23, 24], dtype=np.uint8)
    data7 = np.array([25, 26, 27, 28], dtype=np.uint8)
    data8 = np.array([29, 30, 31, 32], dtype=np.uint8)
    data9 = np.array([33, 34, 35, 36], dtype=np.uint8)
    data10 = np.array([37, 38, 39, 40], dtype=np.uint8)
    data11 = np.array([41, 42, 43, 44], dtype=np.uint8)
    data12 = np.array([45, 46, 47, 48], dtype=np.uint8)
    data13 = np.array([49, 50, 51, 52], dtype=np.uint8)
    data14 = np.array([53, 54, 55, 56], dtype=np.uint8)
    data15 = np.array([57, 58, 59, 60], dtype=np.uint8)
    data16 = np.array([61, 62, 63, 64], dtype=np.uint8)
    data17 = np.array([65, 66, 67, 68], dtype=np.uint8)
    data18 = np.array([69, 70, 71, 72], dtype=np.uint8)
    data19 = np.array([73, 74, 75, 76], dtype=np.uint8)
    data20 = np.array([77, 78, 79, 80], dtype=np.uint8)

    data = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10,
            data11, data12, data13, data14, data15, data16, data17, data18, data19, data20]

    label = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    original_seed = config_get_set_seed(4321)
    assert ds.config.get_seed() == 4321

    dataset = ds.NumpySlicesDataset((data, label), ["data", "label"], num_shards=4, shard_id=2)
    dataset = dataset.batch(batch_size=2)

    save_and_check_dict(dataset, "test_seed_when_distributed_02.npz", False)

    # Restore config setting
    ds.config.set_seed(original_seed)


def test_generator_with_single_numpy():
    """
    Feature: Test GeneratorDataset with single numpy and multi columns when use __getitem__
    Description: Single numpy, tuple numpy with single columns and multi columns
    Expectation: Success
    """

    class get_dataset_generator:
        def __init__(self, value):
            np.random.seed(58)
            self.__value = value

        def __getitem__(self, index):
            return self.__value

        def __len__(self):
            return 20

    def test_generator_one_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            assert (data["data"] == value).all()
            count += 1
        assert count == 20

    # test user define one column
    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_one_column(numpy_1)
    test_generator_one_column(numpy_2)
    test_generator_one_column(numpy_3)
    test_generator_one_column(numpy_4)
    test_generator_one_column(numpy_5)
    test_generator_one_column(numpy_6)
    test_generator_one_column(numpy_7)
    test_generator_one_column(numpy_8)
    test_generator_one_column(numpy_9)
    test_generator_one_column(numpy_10)

    tuple_1 = (numpy_7,)
    dataset_generator = get_dataset_generator(tuple_1)
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == tuple_1[0]).all()
        count += 1
    assert count == 20

    tuple_2 = (numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:2" in str(info.value)

    tuple_4 = (numpy_4, numpy_5, numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_4)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:4" in str(info.value)

    # test user define two column
    def test_generator_two_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:2 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_two_column(numpy_1)
    test_generator_two_column(numpy_2)
    test_generator_two_column(numpy_3)
    test_generator_two_column(numpy_4)
    test_generator_two_column(numpy_5)
    test_generator_two_column(numpy_6)
    test_generator_two_column(numpy_7)
    test_generator_two_column(numpy_8)
    test_generator_two_column(numpy_9)
    test_generator_two_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_two_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    dataset_generator = get_dataset_generator(tuple_2)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_2).all()
        assert (data["label"] == numpy_3).all()
        count += 1
    assert count == 20

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_3)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:2 and number of returned NumPy array is:3" in str(info.value)

    # test user define three column
    def test_generator_three_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False,
                                      num_parallel_workers=number, python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:3 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_three_column(numpy_1)
    test_generator_three_column(numpy_2)
    test_generator_three_column(numpy_3)
    test_generator_three_column(numpy_4)
    test_generator_three_column(numpy_5)
    test_generator_three_column(numpy_6)
    test_generator_three_column(numpy_7)
    test_generator_three_column(numpy_8)
    test_generator_three_column(numpy_9)
    test_generator_three_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_three_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:3 and number of returned NumPy array is:2" in str(info.value)

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    dataset_generator = get_dataset_generator(tuple_3)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_4).all()
        assert (data["label"] == numpy_5).all()
        assert (data["label2"] == numpy_6).all()
        count += 1
    assert count == 20


def test_generator_with_single_numpy_with_next():
    """
    Feature: Test GeneratorDataset with single numpy and multi columns when use __next__
    Description: Single numpy, tuple numpy with single columns and multi columns
    Expectation: Success
    """

    class get_dataset_generator:
        def __init__(self, value):
            np.random.seed(58)
            self.__value = value
            self.__index = 0

        def __next__(self):
            if self.__index >= 20:
                raise StopIteration

            self.__index += 1
            return self.__value

        def __iter__(self):
            self.__index = 0
            return self

        def __len__(self):
            return 20

    def test_generator_one_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            assert (data["data"] == value).all()
            count += 1
        assert count == 20

    # test user define one column
    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_one_column(numpy_1)
    test_generator_one_column(numpy_2)
    test_generator_one_column(numpy_3)
    test_generator_one_column(numpy_4)
    test_generator_one_column(numpy_5)
    test_generator_one_column(numpy_6)
    test_generator_one_column(numpy_7)
    test_generator_one_column(numpy_8)
    test_generator_one_column(numpy_9)
    test_generator_one_column(numpy_10)

    tuple_1 = (numpy_7,)
    dataset_generator = get_dataset_generator(tuple_1)
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == tuple_1[0]).all()
        count += 1
    assert count == 20

    tuple_2 = (numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:2" in str(info.value)

    tuple_3 = (numpy_1, numpy_2)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_3)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:2" in str(info.value)

    tuple_4 = (numpy_4, numpy_5, numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_4)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:4" in str(info.value)

    # test user define two column
    def test_generator_two_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:2 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_two_column(numpy_1)
    test_generator_two_column(numpy_2)
    test_generator_two_column(numpy_3)
    test_generator_two_column(numpy_4)
    test_generator_two_column(numpy_5)
    test_generator_two_column(numpy_6)
    test_generator_two_column(numpy_7)
    test_generator_two_column(numpy_8)
    test_generator_two_column(numpy_9)
    test_generator_two_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_two_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    dataset_generator = get_dataset_generator(tuple_2)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_2).all()
        assert (data["label"] == numpy_3).all()
        count += 1
    assert count == 20

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_3)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:2 and number of returned NumPy array is:3" in str(info.value)

    # test user define three column
    def test_generator_three_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False,
                                      num_parallel_workers=number, python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:3 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_three_column(numpy_1)
    test_generator_three_column(numpy_2)
    test_generator_three_column(numpy_3)
    test_generator_three_column(numpy_4)
    test_generator_three_column(numpy_5)
    test_generator_three_column(numpy_6)
    test_generator_three_column(numpy_7)
    test_generator_three_column(numpy_8)
    test_generator_three_column(numpy_9)
    test_generator_three_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_three_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:3 and number of returned NumPy array is:2" in str(info.value)

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    dataset_generator = get_dataset_generator(tuple_3)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_4).all()
        assert (data["label"] == numpy_5).all()
        assert (data["label2"] == numpy_6).all()
        count += 1
    assert count == 20


def test_generator_with_single_numpy_with_yield():
    """
    Feature: Test GeneratorDataset with single numpy and multi columns when use yield
    Description: Single numpy, tuple numpy with single columns and multi columns
    Expectation: Success
    """

    def get_dataset_generator(value):
        for _ in range(20):
            yield value

    def test_generator_one_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        for data in dataset.create_dict_iterator(output_numpy=True):
            assert (data["data"] == value).all()
            count += 1
        assert count == 20

    # test user define one column
    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_one_column(numpy_1)
    test_generator_one_column(numpy_2)
    test_generator_one_column(numpy_3)
    test_generator_one_column(numpy_4)
    test_generator_one_column(numpy_5)
    test_generator_one_column(numpy_6)
    test_generator_one_column(numpy_7)
    test_generator_one_column(numpy_8)
    test_generator_one_column(numpy_9)
    test_generator_one_column(numpy_10)

    tuple_1 = (numpy_7,)
    dataset_generator = get_dataset_generator(tuple_1)
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == tuple_1[0]).all()
        count += 1
    assert count == 20

    tuple_2 = (numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:2" in str(info.value)

    tuple_3 = (numpy_1, numpy_2)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_3)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:2" in str(info.value)

    tuple_4 = (numpy_4, numpy_5, numpy_6, numpy_7)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_4)
        dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:1 and number of returned NumPy array is:4" in str(info.value)

    # test user define two column
    def test_generator_two_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False, num_parallel_workers=number,
                                      python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:2 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_two_column(numpy_1)
    test_generator_two_column(numpy_2)
    test_generator_two_column(numpy_3)
    test_generator_two_column(numpy_4)
    test_generator_two_column(numpy_5)
    test_generator_two_column(numpy_6)
    test_generator_two_column(numpy_7)
    test_generator_two_column(numpy_8)
    test_generator_two_column(numpy_9)
    test_generator_two_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_two_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    dataset_generator = get_dataset_generator(tuple_2)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_2).all()
        assert (data["label"] == numpy_3).all()
        count += 1
    assert count == 20

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_3)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:2 and number of returned NumPy array is:3" in str(info.value)

    # test user define three column
    def test_generator_three_column(value):
        number = np.random.randint(1, 4)
        process_flag = False
        if number > 1 and number % 2 == 0:
            process_flag = True
        dataset_generator = get_dataset_generator(value)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False,
                                      num_parallel_workers=number, python_multiprocessing=process_flag)
        count = 0
        with pytest.raises(RuntimeError) as info:
            for data in dataset.create_dict_iterator(output_numpy=True):
                print(data)
                count += 1
            assert count == 20
        assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
               "column_names," in str(info.value)
        assert "the size of column_names is:3 and number of returned NumPy array is:1" in str(info.value)

    numpy_1 = np.array(1)
    numpy_2 = np.array([1])
    numpy_3 = np.array([1, 2])
    numpy_4 = np.array([1, 2, 3])
    numpy_5 = np.array([[1], [2]])
    numpy_6 = np.array([[1, 2], [2, 3]])
    numpy_7 = np.array([[1, 2, 3], [2, 3, 4]])
    numpy_8 = np.array([[1], [2], [3]])
    numpy_9 = np.array([[1, 2], [2, 3], [3, 4]])
    numpy_10 = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    test_generator_three_column(numpy_1)
    test_generator_three_column(numpy_2)
    test_generator_three_column(numpy_3)
    test_generator_three_column(numpy_4)
    test_generator_three_column(numpy_5)
    test_generator_three_column(numpy_6)
    test_generator_three_column(numpy_7)
    test_generator_three_column(numpy_8)
    test_generator_three_column(numpy_9)
    test_generator_three_column(numpy_10)
    tuple_1 = (numpy_7,)
    test_generator_three_column(tuple_1)

    tuple_2 = (numpy_2, numpy_3)
    with pytest.raises(RuntimeError) as info:
        dataset_generator = get_dataset_generator(tuple_2)
        dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
        for data in dataset.create_dict_iterator(output_numpy=True):
            print(data["data"])
    assert "the 'source' of 'GeneratorDataset' should return same number of NumPy arrays as specified in " \
           "column_names," in str(info.value)
    assert "the size of column_names is:3 and number of returned NumPy array is:2" in str(info.value)

    tuple_3 = (numpy_4, numpy_5, numpy_6)
    dataset_generator = get_dataset_generator(tuple_3)
    dataset = ds.GeneratorDataset(dataset_generator, ["data", "label", "label2"], shuffle=False)
    count = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        assert (data["data"] == numpy_4).all()
        assert (data["label"] == numpy_5).all()
        assert (data["label2"] == numpy_6).all()
        count += 1
    assert count == 20


@pytest.mark.skip(reason="only for testing stuck scenario")
def test_generator_traceback():
    """
    Feature: GeneratorDataset
    Description: Generator is too slow then main process will log the stack of the stuck process
    Expectation: The stuck locality can be logged
    """

    class SlowDataset:
        def __init__(self):
            self.data = np.random.randint(0, 255, (100, 28, 28, 3), dtype=np.uint8)

        def __getitem__(self, index):
            if index % 10 == 0:
                time.sleep(600)
            return self.data[index]

        def __len__(self):
            return len(self.data)

    dataset = ds.GeneratorDataset(SlowDataset(), column_names=["image"], num_parallel_workers=8)
    for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        pass


def test_generator_split_with_yield():
    """
    Feature: GeneratorDataset
    Description: When GeneratorDataset calls split, it can be split if the input is in yield mode
    Expectation: The dataset is processed as expected
    """
    dataset = ds.GeneratorDataset(generator_1d, ["data"], shuffle=False)
    dataset_train, dataset_val = dataset.split([0.8, 0.2])
    assert dataset_train.get_dataset_size() == 51
    assert dataset_val.get_dataset_size() == 13


def test_generator_split_with_getitem():
    """
    Feature: GeneratorDataset
    Description: When GeneratorDataset calls split, it can be split if the input is in getitem mode
    Expectation: The dataset is processed as expected
    """
    dataset_generator = DatasetGenerator()
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    dataset_train, dataset_val = dataset.split([0.8, 0.2])
    assert dataset_train.get_dataset_size() == 8
    assert dataset_val.get_dataset_size() == 2


def test_generator_split_with_next():
    """
    Feature: GeneratorDataset
    Description: When GeneratorDataset calls split, it can be split if the input is in next mode
    Expectation: The dataset is processed as expected
    """

    class GetDatasetGenerator:
        def __init__(self, data):
            self.__data = data
            self.__count = 0

        def __next__(self):
            if self.__count >= 10:
                raise StopIteration

            self.__count += 1
            return self.__data

        def __iter__(self):
            self.__count = 0
            return self

        def __len__(self):
            return 10

    data_tuple = (np.array([[1, 2, 3], [2, 3, 4]]),)
    dataset_generator = GetDatasetGenerator(data_tuple)
    dataset = ds.GeneratorDataset(dataset_generator, ["data"], shuffle=False)
    dataset_train, dataset_val = dataset.split([0.8, 0.2])
    assert dataset_train.get_dataset_size() == 8
    assert dataset_val.get_dataset_size() == 2


def test_generator_with_next_and_dataset_size_when_iter():
    """
    Feature: GeneratorDataset
    Description: When GeneratorDataset is __next__ and call get_dataset_size in iter
    Expectation: The dataset is processed as expected
    """

    # Iterator as input source
    class Iterator:
        def __init__(self):
            self._index = 0
            self._data = np.random.sample((50, 2))
            self._label = np.random.sample((50, 1))

        def __next__(self):
            if self._index >= len(self._data):
                raise StopIteration

            item = (self._data[self._index], self._label[self._index])
            self._index += 1
            return item

        def __iter__(self):
            self._index = 0
            return self

        def __len__(self):
            return len(self._data)

    data = Iterator()
    dataset = ds.GeneratorDataset(source=data, column_names=["data", "label"])

    count = 0
    for _ in dataset.create_dict_iterator():
        count += 1
    assert count == 50


class FakeData:
    def __init__(self):
        self.input_ids = np.ones((128, 128), dtype=np.int32)
        self.input_mask = np.ones((100, 100), dtype=np.int32)

    def __getitem__(self, index):
        return self.input_ids, self.input_mask

    def __len__(self):
        return 791


def test_generator_multiprocessing_with_fixed_handle():
    """
    Feature: generator op
    Description: generator with multiprocessing and don't leak pipe handle which is used by queue
    Expectation: success
    """

    dataset = ds.GeneratorDataset(FakeData(), ["input_ids", "input_mask"], num_parallel_workers=2)
    assert dataset.get_dataset_size() == 791

    fds = 0
    for i in range(5):
        count = 0
        for item in dataset.create_tuple_iterator(output_numpy=True, num_epochs=1):
            assert item[0].dtype == np.int32
            assert item[0].shape == (128, 128)
            assert len(item) == 2
            count += 1
        assert count == 791

        # wait for the fds handle to be released automatic
        time.sleep(1)

        i += 1
        if i == 1:
            fds = psutil.Process(os.getpid()).num_fds()
            lsof = subprocess.getoutput("lsof -p " + str(os.getpid()) + " | wc -l")
        elif i > 1:
            assert fds >= psutil.Process(os.getpid()).num_fds()
            new_lsof = subprocess.getoutput("lsof -p " + str(os.getpid()) + " | wc -l")
            assert lsof >= new_lsof


def test_generator_with_dynamic_shared_queue():
    """
    Feature: GeneratorDataset
    Description: test GeneratorDataset with dynamic shared memory queue
    Expectation: The dataset is processed as expected
    """

    data = [np.random.random((1024, 1024, 3)).astype(np.float32),  # 12M
            np.random.random((1024, 1024, 3)).astype(np.float32),  # 12M
            np.random.random((1024, 1024, 4)).astype(np.float32),  # 16M
            np.random.random((1024, 1024, 4)).astype(np.float32),  # 16M
            np.random.random((1024, 1024, 5)).astype(np.float32),  # 20M
            np.random.random((1024, 1024, 5)).astype(np.float32)]  # 20M

    class DynamicDataset:
        def __init__(self):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    def map_func(input_data):
        return input_data

    def batch_func(input_data, batch_info):
        return np.array(input_data)

    dataset = ds.GeneratorDataset(DynamicDataset(), column_names=["data"], num_parallel_workers=2,
                                  shuffle=False, max_rowsize=-1)
    dataset = dataset.map(map_func, num_parallel_workers=2, python_multiprocessing=True, max_rowsize=-1)
    dataset = dataset.map(map_func, num_parallel_workers=2, python_multiprocessing=True, max_rowsize=-1)
    dataset = dataset.map(map_func, num_parallel_workers=2, python_multiprocessing=True, max_rowsize=-1)
    dataset = dataset.batch(2, num_parallel_workers=2, per_batch_map=batch_func,
                            python_multiprocessing=True, max_rowsize=-1)

    count = 0
    for sample in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        np.testing.assert_array_equal(sample["data"], np.array(data[count * 2:(count + 1) * 2]))
        count += 1
    assert count == 3


def test_generator_shared_queue_reuse_with_fixed_shape():
    """
    Feature: GeneratorDataset
    Description: Test shared memory reuse with fixed shape data
    Expectation: The dataset is processed as expected
    """

    data = np.random.random((32, 32, 1)).astype(np.uint8)  # 1K

    class FixedDataset:
        def __init__(self):
            self.data = data
            self.len = 60

        def __getitem__(self, index):
            return self.data

        def __len__(self):
            return self.len

    def map_func(input_data):
        return input_data

    def batch_func(input_data, batch_info):
        return np.array(input_data)

    dataset = ds.GeneratorDataset(FixedDataset(), column_names=["data"], num_parallel_workers=2, shuffle=False)
    dataset = dataset.map(map_func, num_parallel_workers=2, python_multiprocessing=True)
    dataset = dataset.map(map_func, num_parallel_workers=2, python_multiprocessing=True)
    dataset = dataset.map(map_func, num_parallel_workers=2, python_multiprocessing=True)
    dataset = dataset.batch(2, num_parallel_workers=2, per_batch_map=batch_func, python_multiprocessing=True)

    count = 0
    for sample in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        np.testing.assert_array_equal(sample["data"], np.array([data, data]))
        count += 1
    assert count == 30


def test_generator_shared_queue_reuse_with_dynamic_shape():
    """
    Feature: GeneratorDataset
    Description: Test shared memory reuse with dynamic shape data
    Expectation: The dataset is processed as expected
    """

    data = [np.random.random((32, 32, i + 1)).astype(np.uint8) for i in range(30)]

    class DynamicDataset:
        def __init__(self):
            self.data = data

        def __getitem__(self, index):
            return self.data[index // 2]

        def __len__(self):
            return len(self.data) * 2

    def map_func(input_data):
        return input_data

    def batch_func(input_data, batch_info):
        return np.array(input_data)

    dataset = ds.GeneratorDataset(DynamicDataset(), column_names=["data"], num_parallel_workers=2, shuffle=False)
    dataset = dataset.map(map_func, num_parallel_workers=2, python_multiprocessing=True)
    dataset = dataset.map(map_func, num_parallel_workers=2, python_multiprocessing=True)
    dataset = dataset.map(map_func, num_parallel_workers=2, python_multiprocessing=True)
    dataset = dataset.batch(2, num_parallel_workers=2, per_batch_map=batch_func, python_multiprocessing=True)

    count = 0
    for sample in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        np.testing.assert_array_equal(sample["data"], np.array([data[count], data[count]]))
        count += 1
    assert count == 30


def test_generator_shared_queue_reuse_with_empty_numpy():
    """
    Feature: GeneratorDataset
    Description: Test shared memory reuse with empty numpy and multiple columns
    Expectation: The dataset is processed as expected
    """

    data = [(np.random.random((15, 25)), np.random.random((22, 13)), np.random.random((17, 23))),  # (data, data, data)
            (np.array([]), np.random.random((29, 77)), np.random.random((64, 18))),  # (empty, data, data)
            (np.random.random((7, 16)), np.array([]), np.random.random((9, 51))),  # (data, empty, data)
            (np.random.random((3, 111)), np.random.random((5, 27)), np.array([])),  # (data, data, empty)
            (np.array([]), np.array([]), np.random.random((37, 26))),  # (empty, empty, data)
            (np.array([]), np.random.random((58, 76)), np.array([])),  # (empty, data, empty)
            (np.random.random((91, 43)), np.array([]), np.array([])),  # (data, empty, empty)
            (np.array([]), np.array([]), np.array([]))]  # (empty, empty, empty)

    class DynamicDataset:
        def __init__(self):
            self.data = data

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    def map_func(input_data):
        return input_data

    dataset = ds.GeneratorDataset(DynamicDataset(), column_names=["data1", "data2", "data3"],
                                  num_parallel_workers=2, shuffle=False)
    dataset = dataset.map(map_func, python_multiprocessing=True)
    dataset = dataset.map(map_func, python_multiprocessing=True)
    dataset = dataset.map(map_func, python_multiprocessing=True)

    count = 0
    num_columns = 3
    for sample in dataset.create_tuple_iterator(output_numpy=True, num_epochs=1):
        for i in range(num_columns):
            np.testing.assert_array_equal(sample[i], data[count][i])
        count += 1
    assert count == 8


def test_generator_with_invalid_max_row_size():
    """
    Feature: GeneratorDataset
    Description: test GeneratorDataset with invalid max_rowsize when using shared memory
    Expectation: Raise errors as expected
    """

    with pytest.raises(ValueError) as e:
        _ = ds.GeneratorDataset(DatasetGenerator(), column_names=["data"], num_parallel_workers=2,
                                shuffle=False, max_rowsize=-2)
    assert "not within the required interval of [-1, 2147483647]" in str(e.value)


def test_generator_with_seed_and_multiprocessing_mode():
    """
    Feature: GeneratorDataset
    Description: test GeneratorDataset with seed in multiprocessing mode
    Expectation: SUCCESS
    """
    origin_seed = ds.config.get_seed()
    ds.config.set_seed(1234)

    expected_data = [990, 931, 797, 706, 452, 435, 120, 549, 8, 863, 93, 607, 933,
                     482, 966, 401, 962, 153, 827, 978, 597, 725, 36, 358, 688, 739,
                     710, 662, 86, 804]

    expected_data2 = [1989, 1930, 1796, 1705, 1451, 1434, 1119, 1548, 1007, 1862, 1606,
                      1092, 1481, 1932, 1400, 1965, 1152, 1961, 1977, 1826, 1596, 1724,
                      1035, 1357, 1687, 1738, 1709, 1661, 1085, 1803]

    # Random-accessible object as input source
    class RandintDataset:
        def __init__(self):
            pass

        def __getitem__(self, index):
            return np.array(random.randint(1, 1000)), np.array(1)

        def __len__(self):
            return 10

    loader = RandintDataset()
    dataset = ds.GeneratorDataset(source=loader, column_names=["data", "label"], num_parallel_workers=2)

    def add_column(data):
        return data, np.array(random.randint(1000, 2000))

    dataset = dataset.map(add_column, input_columns=["data"], output_columns=["data", "data2"], num_parallel_workers=2,
                          python_multiprocessing=True)

    epoch = 3
    dataset_iter = dataset.create_dict_iterator(num_epochs=epoch)
    index = 0
    for _ in range(epoch):
        for item in dataset_iter:
            assert item["data"] == expected_data[index]
            assert item["data2"] == expected_data2[index]
            index += 1

    ds.config.set_seed(origin_seed)


def test_generator_with_generator_object_iterated_multi_times():
    """
    Feature: GeneratorDataset
    Description: test GeneratorDataset with generator object iterated in multi times
    Expectation: SUCCESS
    """

    # Generator
    def my_generator(start, end):
        for i in range(start, end):
            yield i

    expected = [x for x in my_generator(3, 6)]

    dataset = ds.GeneratorDataset(source=my_generator(3, 6), column_names=["data"])

    assert dataset.get_dataset_size() == 3
    assert dataset.output_shapes() == [[]]
    assert dataset.output_types() == [np.int64]

    count = 0
    for _ in range(5):
        index = 0
        for d in dataset.create_tuple_iterator(output_numpy=True):
            assert len(d) == 1
            assert d[0] == expected[index]
            index += 1
            count += 1
        assert index == 3
    assert count == 15

    epochs = 3
    dataset_iter = dataset.create_tuple_iterator(output_numpy=True, num_epochs=epochs)
    count = 0
    for _ in range(epochs):
        index = 0
        for d in dataset_iter:
            assert len(d) == 1
            assert d[0] == expected[index]
            index += 1
            count += 1
        assert index == 3
    assert count == 9


@pytest.mark.parametrize("num_epochs", (-1, 1, 10))
def test_release_generator_dataset_iter(num_epochs):
    """
    Feature: GeneratorDataset
    Description: Test memory collection of GeneratorDataset
    Expectation: After destructing all the instance created, the memory should be released
    """
    original_prefetch_size = ds.config.get_prefetch_size()
    ds.config.set_prefetch_size(1)

    init_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024

    class Iterable:
        def __init__(self):
            self.a = [np.ones((2048 * 2048 * 3), dtype=np.int64),  # 96M
                      np.ones((2048 * 2048 * 3 * 2), dtype=np.int64),  # 192M
                      np.ones((2048 * 2048 * 3 * 3), dtype=np.int64),  # 288M
                      np.ones((2048 * 2048 * 3 * 4), dtype=np.int64),  # 384M
                      np.ones((2048 * 2048 * 3 * 5), dtype=np.int64)]  # 480M
            self.b = [np.ones((1024 * 1024 * 5), dtype=np.int64),  # 40M
                      np.ones((1024 * 1024 * 5 * 2), dtype=np.int64),  # 80M
                      np.ones((1024 * 1024 * 5 * 3), dtype=np.int64),  # 120M
                      np.ones((1024 * 1024 * 5 * 4), dtype=np.int64),  # 160M
                      np.ones((1024 * 1024 * 5 * 5), dtype=np.int64)]  # 200M
            self.len = len(self.a)

        def __getitem__(self, index):
            return self.a[4 - index], self.b[4 - index]

        def __len__(self):
            return self.len

    # initialize user defined dataset
    data = Iterable()
    data_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    assert (data_memory - init_memory) > 1990

    # initialize GeneratorDataset
    dataset = ds.GeneratorDataset(source=data, column_names=["data", "label"], shuffle=False)
    dataset_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    assert (dataset_memory - data_memory) < 3

    # initialize Iterator
    ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=num_epochs)
    iterator_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    assert (iterator_memory - dataset_memory) < 3

    # process and fetch data
    epochs = 1 if num_epochs == -1 else num_epochs
    item = None
    for _ in range(epochs):
        for item in ds_iter:
            break

    process_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    assert (process_memory - iterator_memory) < 3

    # destruct all the instance
    del item
    del ds_iter
    del dataset
    del data

    # all the memory should be released
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
    assert (end_memory - init_memory) < 3

    ds.config.set_prefetch_size(original_prefetch_size)


@pytest.mark.parametrize('num_samples', (None, 4))
@pytest.mark.parametrize('shuffle', (None, False, True))
def test_generator_dataset_getitem_success_distributed_sampler(num_samples, shuffle):
    """
    Feature: GeneratorDataset random access
    Description: Test combinations of GeneratorDataset parameters [shuffle/num_samples]
    Expectation: SUCCESS
    """
    small_dataset = DatasetGeneratorSmall()
    origin_seed = ds.config.get_seed()

    ds.config.set_seed(200)
    dataset = ds.GeneratorDataset(small_dataset, column_names=["col1"], shuffle=shuffle, num_samples=num_samples,
                                  num_shards=3, shard_id=0)
    index = 0
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert item == dataset[index]
        index += 1
    ds.config.set_seed(origin_seed)


@pytest.mark.parametrize('num_samples', (None, 4))
@pytest.mark.parametrize('shuffle', (None, False, True))
def test_generator_dataset_getitem_success_other_sampler(num_samples, shuffle):
    """
    Feature: GeneratorDataset random access
    Description: Test combinations of GeneratorDataset parameters [shuffle/num_samples]
    Expectation: SUCCESS
    """
    small_dataset = DatasetGeneratorSmall()
    origin_seed = ds.config.get_seed()

    ds.config.set_seed(200)
    dataset = ds.GeneratorDataset(small_dataset, column_names=["col1"], shuffle=shuffle, num_samples=num_samples,
                                  num_shards=None, shard_id=None)
    index = 0
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert item == dataset[index]
        index += 1
    ds.config.set_seed(origin_seed)


def test_generator_dataset_getitem_success_sampler():
    """
    Feature: GeneratorDataset random access
    Description: Test combinations of GeneratorDataset parameters [shuffle/sampler]
    Expectation: SUCCESS
    """
    small_dataset = DatasetGeneratorSmall()
    origin_seed = ds.config.get_seed()
    ds.config.set_seed(200)

    weights = [0.9, 0.01, 0.4, 0.8, 0.1, 0.3]
    sampler = ds.WeightedRandomSampler(weights, 4)
    dataset = ds.GeneratorDataset(small_dataset, column_names=["col1"], shuffle=None, sampler=sampler)
    index = 0
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert item == dataset[index]
        index += 1
    ds.config.set_seed(origin_seed)

    # Test the source dataset is a list object.
    dataset = ds.GeneratorDataset([1, 2, 3, 4, 5], column_names=["col1"], shuffle=False)
    index = 0
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert item[0] == dataset[index]
        index += 1


def test_generator_dataset_getitem_schema():
    """
    Feature: GeneratorDataset random access
    Description: Test combinations of GeneratorDataset parameters [schema]
    Expectation: SUCCESS
    """
    small_dataset = DatasetGeneratorSmall()
    schema = ds.Schema()
    schema.add_column("col1", de_type=mstype.int32)
    dataset = ds.GeneratorDataset(small_dataset, shuffle=False, schema=schema)
    index = 0
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert item == dataset[index]
        index += 1


def test_generator_dataset_getitem_two_level_pipeline():
    """
    Feature: GeneratorDataset random access
    Description: Verify GeneratorDataset random access on two level pipeline
    Expectation: SUCCESS
    """
    source_dataset = DatasetGeneratorTwoLevelPipeline()
    dataset = ds.GeneratorDataset(source_dataset, column_names=["col1"], shuffle=False)
    index = 0
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert item == dataset[index]
        index += 1


def test_generator_dataset_getitem_exception():
    """
    Feature: GeneratorDataset random access exception
    Description: Test the exception to GeneratorDataset random access
    Expectation: Success throw exception
    """
    small_dataset = DatasetGeneratorSmall()

    # Test source dataset do not have "__getitem__" function.
    with pytest.raises(RuntimeError, match="Dataset don't support randomized access."):
        dataset = ds.GeneratorDataset(source=generator_1d, column_names=["col1"], shuffle=False)
        _ = dataset[0]

    # Test the number of input indexes exceeds the number of samples
    dataset = ds.GeneratorDataset(source=small_dataset, column_names=["col1"], shuffle=False)
    with pytest.raises(RuntimeError, match=re.escape("Index [8] exceeded the number of data samples.")):
        _ = dataset[8]

    # Test the input index is an abnormal value.
    dataset = ds.GeneratorDataset(source=small_dataset, column_names=["col1"], shuffle=False)
    err_info = "Argument index with value x is not of type [<class 'int'>, <class 'numpy.number'>], " \
               "but got <class 'str'>."
    with pytest.raises(TypeError, match=re.escape(err_info)):
        _ = dataset["x"]

    # Test the input index is a negative number
    with pytest.raises(RuntimeError) as err_info:
        dataset = ds.GeneratorDataset(source=small_dataset, column_names=["col1"], shuffle=False)
        _ = dataset[-1]
        assert "Index [-1] can not be a negative number." in str(err_info.value)


class SimpleBatchSampler:
    def __init__(self):
        self.indices = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    def __iter__(self):
        return iter(self.indices)


class SimpleDataset:
    def __init__(self):
        self.data = [np.array([i], dtype=np.int32) for i in range(10)]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


@pytest.mark.parametrize("debug_mode", (False, True))
def test_generator_with_batch_sampler(debug_mode):
    """
    Feature: BatchSampler
    Description: Test GeneratorDataset with batch_sampler
    Expectation: Result is as expected
    """

    original_mode = ds.config.get_debug_mode()
    ds.config.set_debug_mode(debug_mode)

    dataset = ds.GeneratorDataset(SimpleDataset(), column_names=["data"], batch_sampler=SimpleBatchSampler())

    assert dataset.get_dataset_size() == 5
    assert dataset.output_shapes() == [[2, 1]]
    assert dataset.output_types() == [np.int32]
    assert dataset.get_col_names() == ["data"]
    assert dataset.get_batch_size() == -1

    expected_res = [[[0], [1]], [[2], [3]], [[4], [5]], [[6], [7]], [[8], [9]]]
    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        np.testing.assert_array_equal(data["data"], np.array(expected_res[i], dtype=np.int32))

    ds.config.set_debug_mode(original_mode)


def test_generator_with_batch_sampler_and_split():
    """
    Feature: BatchSampler
    Description: Test GeneratorDataset with batch_sampler and split
    Expectation: Result is as expected
    """

    dataset = ds.GeneratorDataset(SimpleDataset(), column_names=["data"], batch_sampler=SimpleBatchSampler())
    train_dataset, _ = dataset.split(sizes=[3, 2], randomize=False)

    expected_res = [[[0], [1]], [[2], [3]], [[4], [5]]]
    for i, data in enumerate(train_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        np.testing.assert_array_equal(data["data"], np.array(expected_res[i], dtype=np.int32))


@pytest.mark.parametrize("debug_mode", (False, True))
def test_generator_with_collate_fn(debug_mode):
    """
    Feature: BatchSampler
    Description: Test GeneratorDataset with batch_sampler and collate_fn
    Expectation: Result is as expected
    """

    original_mode = ds.config.get_debug_mode()
    ds.config.set_debug_mode(debug_mode)

    class DictDataset:
        def __init__(self):
            self.data = [({"data": np.array([i], dtype=np.int32)}, 1) for i in range(10)]

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    def collate_fn(data, label):
        data = [sample["data"] for sample in data]
        return np.stack(data, axis=0), np.stack(label, axis=0)

    dataset = ds.GeneratorDataset(DictDataset(), column_names=["data", "label"],
                                  batch_sampler=SimpleBatchSampler(), collate_fn=collate_fn)

    expected_res = [[[0], [1]], [[2], [3]], [[4], [5]], [[6], [7]], [[8], [9]]]
    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        np.testing.assert_array_equal(data["data"], np.array(expected_res[i], dtype=np.int32))
        np.testing.assert_array_equal(data["label"], np.array(1))

    ds.config.set_debug_mode(original_mode)


def test_generator_with_batch_sampler_in_recovery_mode():
    """
    Feature: BatchSampler
    Description: Test GeneratorDataset with batch_sampler and collate_fn in recovery mode
    Expectation: Result is as expected
    """

    class RandomBatchSampler:
        def __init__(self):
            self.indices = [[12, 1, 5], [10], [11, 8, 6, 7], [2, 0], [4], [3, 9]]
            self.generator = np.random.default_rng(0)

        def __iter__(self):
            self.generator.shuffle(self.indices)
            return iter(self.indices)

    class MyDataset:
        def __init__(self):
            self.data = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']

        def __getitem__(self, index):
            return self.data[index]

        def __len__(self):
            return len(self.data)

    def collate_fn(batch):
        return {"data": np.stack(batch, 0)}

    def get_value_from_dict(data):
        return data["data"]

    dataset1 = ds.GeneratorDataset(MyDataset(), column_names=["data"],
                                   batch_sampler=RandomBatchSampler(), collate_fn=collate_fn)
    dataset2 = ds.GeneratorDataset(MyDataset(), column_names=["data"],
                                   batch_sampler=RandomBatchSampler(), collate_fn=collate_fn)
    dataset = dataset1 + dataset2
    dataset = dataset.map(get_value_from_dict, input_columns=["data"])

    dataset_size = dataset.get_dataset_size()
    assert dataset_size == 12

    num_epochs = 3

    # iterate for the first time to save expected result
    iterator1 = dataset.create_tuple_iterator(output_numpy=True, num_epochs=num_epochs)
    res1 = []
    for _ in range(num_epochs):
        for data in iterator1:
            res1.append(data[0].tolist())

    # recovery from the second epoch and check if the result is as expected
    init_step = dataset_size
    dataset.set_init_step(init_step)
    iterator2 = dataset.create_tuple_iterator(output_numpy=True, num_epochs=num_epochs)
    res2 = []
    for _ in range(num_epochs - init_step // dataset_size):
        for data in iterator2:
            res2.append(data[0].tolist())

    assert res1[init_step:] == res2


@pytest.mark.parametrize("kwargs", ({"num_samples": 1}, {"shuffle": True},
                                    {"num_shards": 8, "shard_id": 0},
                                    {"sampler": ds.RandomSampler()}))
def test_generator_batch_sampler_exclusive_with_other_param(kwargs):
    """
    Feature: BatchSampler
    Description: Test GeneratorDataset with batch_sampler and some exclusive params
    Expectation: Raise error as expected
    """
    with pytest.raises(ValueError) as e:
        _ = ds.GeneratorDataset(source=DatasetGenerator(), column_names=["data"],
                                batch_sampler=SimpleBatchSampler(), **kwargs)
    assert ("batch_sampler is mutually exclusive with num_samples, shuffle, num_shards, "
            "shard_id and sampler") in str(e.value)


def test_generator_invalid_batch_sampler():
    """
    Feature: BatchSampler
    Description: Test GeneratorDataset with invalid batch_sampler
    Expectation: Raise error as expected
    """
    with pytest.raises(TypeError) as e:
        _ = ds.GeneratorDataset(source=DatasetGenerator(), column_names=["data"],
                                batch_sampler=1)
    assert "batch_sampler should have __getitem__ or __iter__ method" in str(e.value)

    with pytest.raises(ValueError) as e:
        _ = ds.GeneratorDataset(source=IterableDataset(), column_names=["data"],
                                batch_sampler=SimpleBatchSampler())
    assert "batch_sampler is not supported if source does not have attribute '__getitem__'" in str(e.value)

    with pytest.raises(RuntimeError) as e:
        dataset = ds.GeneratorDataset(source=DatasetGenerator(), column_names=["data"],
                                      batch_sampler=[1, 2, 3])
        for _ in dataset.create_dict_iterator(num_epochs=1):
            pass
    assert "Batch sampler should return a list, but got an integer" in str(e.value)

    with pytest.raises(RuntimeError) as e:
        dataset = ds.GeneratorDataset(source=DatasetGenerator(), column_names=["data"],
                                      batch_sampler=[["1"], ["2"], ["3"]])
        for _ in dataset.create_dict_iterator(num_epochs=1):
            pass
    assert "Python sampler should return index of type integer" in str(e.value)


def test_generator_invalid_collate_fn():
    """
    Feature: BatchSampler
    Description: Test GeneratorDataset with invalid collate_fn
    Expectation: Raise error as expected
    """
    with pytest.raises(ValueError) as e:
        _ = ds.GeneratorDataset(source=DatasetGenerator(), column_names=["data"],
                                collate_fn=lambda batch: batch)
    assert "collate_fn can be specified only when batch_sampler is set" in str(e.value)

    with pytest.raises(TypeError) as e:
        _ = ds.GeneratorDataset(source=DatasetGenerator(), column_names=["data"],
                                batch_sampler=SimpleBatchSampler(), collate_fn=[])
    assert "collate_fn should be callable" in str(e.value)


def test_generator_dataset_with_parallel_convert():
    """
    Feature: Parallel convert tensor
    Description: Test the data on parallel convert mode
    Expectation: Keep the data the same as in normal iterations
    """
    ds.config.set_iterator_mode(parallel_convert=True)

    datataset = ds.GeneratorDataset(generator_1d, ["data"])

    i = 0
    for item in datataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        golden = np.array([i])
        np.testing.assert_array_equal(item["data"], golden)
        i = i + 1

    ds.config.set_iterator_mode(parallel_convert=False)


def test_generator_dataset_with_parallel_convert_break():
    """
    Feature: Parallel convert tensor
    Description: Test Interrupt iteration on parallel convert mode
    Expectation: Iteration value is the same as expected
    """
    ds.config.set_iterator_mode(parallel_convert=True)

    dataset = ds.GeneratorDataset(generator_1d, ["data"])

    data_iter = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)

    index = 0
    for item in data_iter:
        golden = np.array([index])
        np.testing.assert_array_equal(item["data"], golden)
        if index == 2:
            index += 1
            break
        index += 1

    for item in data_iter:
        golden = np.array([index])
        np.testing.assert_array_equal(item["data"], golden)
        index += 1

    ds.config.set_iterator_mode(parallel_convert=False)


def test_generator_dataset_with_parallel_convert_exception():
    """
    Feature: The exception on parallel convert tensor
    Description: Test the data on parallel convert mode with exception
    Expectation: Success throw exception
    """
    ds.config.set_iterator_mode(parallel_convert=True)

    def generator_function():
        raise RuntimeError("Exceptional error.")

    dataset = ds.GeneratorDataset(generator_function, ["data"])

    with pytest.raises(RuntimeError) as err_info:
        for _ in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass
        assert "Exceptional error." in str(err_info.value)

    ds.config.set_iterator_mode(parallel_convert=False)


def test_generator_dataset_debug_mode():
    """
    Feature: GeneratorDataset random access
    Description: Test GeneratorDataset on debug_mode
    Expectation: SUCCESS
    """
    small_dataset = DatasetGeneratorSmall()
    origin_seed = ds.config.get_seed()

    ds.config.set_debug_mode(True)
    ds.config.set_seed(200)
    dataset = ds.GeneratorDataset(small_dataset, column_names=["col1"], shuffle=False, num_samples=None,
                                  num_shards=4, shard_id=3)
    index = 0
    for item in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
        assert item == dataset[index]
        index += 1
    ds.config.set_seed(origin_seed)
    ds.config.set_debug_mode(False)


def test_perf_do_copy_parameter():
    """
    Feature: Performance comparison of the do_copy parameter
    Description: Testing do_copy parameter False outperforms True
    Expectation: SUCCESS
    """
    source = CustomizedData(256, 1, 8193)
    dataset = ds.GeneratorDataset(source, ["input_ids"], shuffle=False)

    start_time = time.time()
    for _ in range(20):
        for _ in dataset.create_dict_iterator(output_numpy=False, do_copy=False):
            pass
    do_copy_false_time = (time.time() - start_time) / 20

    dataset1 = ds.GeneratorDataset(source, ["input_ids"], shuffle=False)
    start_time1 = time.time()
    for _ in range(20):
        for _ in dataset1.create_dict_iterator(output_numpy=False, do_copy=True):
            pass
    do_copy_true_time = (time.time() - start_time1) / 20
    assert do_copy_false_time < do_copy_true_time


if __name__ == "__main__":
    test_generator_0()
    test_generator_1()
    test_generator_2()
    test_generator_3()
    test_generator_4()
    test_generator_5()
    test_generator_6()
    test_generator_7()
    test_generator_8()
    test_generator_9()
    test_generator_10()
    test_generator_11()
    test_generator_12()
    test_generator_13()
    test_generator_14()
    test_generator_15()
    test_generator_16()
    test_generator_17()
    test_generator_18()
    test_generator_19()
    test_generator_20()
    test_generator_error_1()
    test_generator_error_2()
    test_generator_error_4()
    test_generator_sequential_sampler()
    test_generator_random_sampler()
    test_generator_distributed_sampler()
    test_generator_num_samples()
    test_generator_num_samples_underflow()
    test_generator_schema()
    test_generator_dataset_size_0()
    test_generator_dataset_size_1()
    test_generator_dataset_size_2()
    test_generator_dataset_size_3()
    test_generator_dataset_size_4()
    test_generator_dataset_size_5()
    test_explicit_deepcopy()
    test_func_generator_dataset_005()
    test_func_generator_dataset_with_zip_source()
    test_generator_mixed_operator()
    test_generator_single_input_0()
    test_generator_single_input_1()
    test_generator_single_input_2()
    test_generator_single_input_3()
    test_generator_single_input_4()
    test_generator_single_input_5()
    test_generator_single_input_6()
    test_generator_one_dimensional_numpy_input()
    test_generator_with_seed_5489_when_dist()
    test_generator_with_set_seed_when_dist()
    test_generator_with_single_numpy()
    test_generator_with_single_numpy_with_next()
    test_generator_with_single_numpy_with_yield()
    test_generator_traceback()
    test_generator_split_with_yield()
    test_generator_split_with_getitem()
    test_generator_split_with_next()
    test_generator_with_next_and_dataset_size_when_iter()
    test_generator_multiprocessing_with_fixed_handle()
    test_generator_with_dynamic_shared_queue()
    test_generator_shared_queue_reuse_with_fixed_shape()
    test_generator_shared_queue_reuse_with_dynamic_shape()
    test_generator_shared_queue_reuse_with_empty_numpy()
    test_generator_with_invalid_max_row_size()
    test_generator_with_generator_object_iterated_multi_times()
    test_generator_with_seed_and_multiprocessing_mode()
    test_release_generator_dataset_iter(1)
    test_generator_dataset_getitem_success_distributed_sampler(None, True)
    test_generator_dataset_getitem_success_other_sampler(None, True)
    test_generator_dataset_getitem_success_sampler()
    test_generator_dataset_getitem_schema()
    test_generator_dataset_getitem_two_level_pipeline()
    test_generator_dataset_getitem_exception()
    test_generator_with_batch_sampler(False)
    test_generator_with_batch_sampler_and_split()
    test_generator_with_collate_fn(False)
    test_generator_with_batch_sampler_in_recovery_mode()
    test_generator_batch_sampler_exclusive_with_other_param({"num_samples": 1})
    test_generator_invalid_batch_sampler()
    test_generator_invalid_collate_fn()
    test_generator_dataset_with_parallel_convert()
    test_generator_dataset_with_parallel_convert_break()
    test_generator_dataset_with_parallel_convert_exception()
    test_generator_dataset_debug_mode()
    test_perf_do_copy_parameter()
