# Copyright 2024 Huawei Technologies Co., Ltd
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
import time
import pytest

import numpy as np

from mindspore.mindrecord import FileWriter
import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.audio as audio
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms


apple_jpg = "../data/dataset/apple.jpg"
MNIST_DIR = "../data/dataset/testMnistData"
WIKI_DIR = '../data/dataset/testWikiText'
GTZAN_DIR = "../data/dataset/testGTZANData"
TEXT_FILE = "../data/dataset/testTextFileDataset/1.txt"


def test_dataset_with_independent_process():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "true"
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self._label = np.zeros((1), dtype=np.uint32)
            self.image = np.fromfile(apple_jpg, dtype=np.int8)

        def __getitem__(self, index):
            return self.image, self._label, np.array("./abcdefg.jpg")

        def __len__(self):
            return 10

    def PyFunc(img):
        img = vision.Decode()(img)
        img = vision.Resize((224, 224))(img)
        img = vision.Rescale(1.0 / 255.0, 0.0)(img)
        img = vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(img)
        return img

    loader = RandomAccessDataset()
    dataset = ds.GeneratorDataset(source=loader, column_names=["data", "label", "file_name"])
    dataset = dataset.map(operations=PyFunc, input_columns=["data"], num_parallel_workers=4,
                          python_multiprocessing=True)
    dataset = dataset.batch(batch_size=2)

    count = 0
    start = time.time()
    avg = 0
    epochs = 3
    epoch = 0
    assert dataset.get_dataset_size() == 5
    assert dataset.output_shapes() == [[2, 224, 224, 3], [2, 1], [2,]]
    assert dataset.output_types()[0:2] == [np.float32, np.uint32]
    assert dataset.get_col_names() == ["data", "label", "file_name"]
    assert dataset.get_batch_size() == 2
    ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    for _ in range(epochs):
        for item in ds_iter:
            assert len(item["file_name"]) == 2
            assert item["file_name"][0] == np.array("./abcdefg.jpg")
            if count > 1:
                cost = time.time() - start
                avg += cost
                print("epoch: {}, time cost: {}, count: {}, avg: {}".format(epoch, cost, count, avg / (count - 1)),
                      flush=True)
            count += 1
            start = time.time()
        epoch += 1
    assert count == 15
    assert epoch == 3
    del os.environ["MS_INDEPENDENT_DATASET"]


def test_dataset_with_independent_process_dynamic_shape():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with dynamic shape
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "true"
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self, diff_shapes):
            self._label = np.zeros((1), dtype=np.float32)
            self.image = np.fromfile(apple_jpg, dtype=np.int8)
            self.sizes = diff_shapes

        def __getitem__(self, index):
            img = vision.Decode()(self.image)
            img = vision.Resize(self.sizes[index % 5])(img)
            return img, self._label

        def __len__(self):
            return 10

    diff_shapes = [(548, 506), (778, 578), (1024, 700), (1358, 734), (1570, 882)]
    loader = RandomAccessDataset(diff_shapes)
    dataset = ds.GeneratorDataset(source=loader, column_names=["data", "label"])

    count = 0
    start = time.time()
    avg = 0
    epochs = 3
    epoch = 0
    shapes_count = [0, 0, 0, 0, 0]
    assert dataset.get_dataset_size() == 10
    shapes = dataset.output_shapes()
    assert tuple(shapes[0][0:2]) in diff_shapes
    assert shapes[1] == [1]
    assert dataset.output_types() == [np.uint8, np.float32]
    assert dataset.get_col_names() == ["data", "label"]
    assert dataset.get_batch_size() == 1
    ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    for _ in range(epochs):
        for item in ds_iter:
            shapes_count[diff_shapes.index(item["data"].shape[0:2])] += 1
            if count > 1:
                cost = time.time() - start
                avg += cost
                print("epoch: {}, time cost: {}, count: {}, avg: {}".format(epoch, cost, count, avg / (count - 1)),
                      flush=True)
            count += 1
            start = time.time()
        epoch += 1
    assert len(np.unique(np.array(shapes_count))) == 1
    assert shapes_count[0] == 6
    assert sum(shapes_count) == 30
    assert count == 30
    assert epoch == 3
    del os.environ["MS_INDEPENDENT_DATASET"]


def test_dataset_with_independent_process_train_and_eval():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with train and eval dataset
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "true"
    class TrainUDFDataset:
        def __init__(self):
            self._label = np.zeros((1), dtype=np.uint32)
            self.image = np.fromfile(apple_jpg, dtype=np.int8)

        def __getitem__(self, index):
            return self.image, self._label

        def __len__(self):
            return 10

    class EvalUDFDataset:
        def __init__(self):
            self._label = np.zeros((2), dtype=np.float32)
            self.image = np.fromfile(apple_jpg, dtype=np.int8)

        def __getitem__(self, index):
            return self.image, self._label

        def __len__(self):
            return 4

    def TrainPyFunc(img):
        img = vision.Decode()(img)
        img = vision.Resize((224, 224))(img)
        img = vision.Rescale(1.0 / 255.0, 0.0)(img)
        img = vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(img)
        return img

    def EvalPyFunc(img):
        img = vision.Decode()(img)
        img = vision.Resize((187, 187))(img)
        return img

    dataset = ds.GeneratorDataset(source=TrainUDFDataset(), column_names=["data", "label"])
    dataset = dataset.map(operations=TrainPyFunc, input_columns=["data"], num_parallel_workers=4,
                          python_multiprocessing=True)
    dataset = dataset.batch(batch_size=2)

    dataset2 = ds.GeneratorDataset(source=EvalUDFDataset(), column_names=["data", "label"], shuffle=False)
    dataset2 = dataset2.map(operations=EvalPyFunc, input_columns=["data"], num_parallel_workers=4,
                            python_multiprocessing=True)

    count = 0
    start = time.time()
    avg = 0
    epochs = 3
    epoch = 0
    assert dataset.get_dataset_size() == 5
    assert dataset.output_shapes() == [[2, 224, 224, 3], [2, 1]]
    assert dataset.output_types() == [np.float32, np.uint32]
    assert dataset.get_col_names() == ["data", "label"]
    assert dataset.get_batch_size() == 2
    assert dataset2.get_dataset_size() == 4
    assert dataset2.output_shapes() == [[187, 187, 3], [2]]
    assert dataset2.output_types() == [np.uint8, np.float32]
    assert dataset2.get_col_names() == ["data", "label"]
    assert dataset2.get_batch_size() == 1
    ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    # train process
    for _ in range(epochs):
        for item in ds_iter:
            assert item['data'].shape == (2, 224, 224, 3)
            assert item['data'].dtype == np.float32
            assert item['label'][0] == np.zeros((1), dtype=np.uint32)
            assert item['label'].shape == (2, 1)
            assert item['label'].dtype == np.uint32
            if count > 1:
                cost = time.time() - start
                avg += cost
                print("epoch: {}, time cost: {}, count: {}, avg: {}".format(epoch, cost, count, avg / (count - 1)),
                      flush=True)
            count += 1
            start = time.time()

            # eval process
            if count % 100 == 0:
                ds_iter2 = dataset2.create_dict_iterator(output_numpy=True, num_epochs=1)
                count2 = 0
                for item2 in ds_iter2:
                    assert item2['data'].shape == (187, 187, 3)
                    assert item2['data'].dtype == np.uint8
                    assert item['label'][0] == np.zeros((2), dtype=np.float32)
                    assert item['label'].shape == (1)
                    assert item['label'].dtype == np.float32
                    print("count2: {}".format(count2), flush=True)
                    count2 += 1
                assert count2 == 4
        epoch += 1
    assert count == 15
    assert epoch == 3
    del os.environ["MS_INDEPENDENT_DATASET"]


def print_psutil(name):
    print("============== {} =============".format(name), flush=True)
    os.system("ps -ef | grep python")


def test_dataset_with_independent_process_two_stage_pipeline():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with two stage pipeline
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "true"
    class FristUDFDataset:
        def __init__(self):
            self._image = np.fromfile(apple_jpg, dtype=np.int8)
            self._label = np.zeros((1), dtype=np.uint32)

        def __getitem__(self, index):
            return self._image, self._label

        def __len__(self):
            return 10

    def FirstPyFunc(img):
        img = vision.Decode()(img)
        img = vision.Resize((300, 300))(img)
        return img

    first_dataset = ds.GeneratorDataset(source=FristUDFDataset(), column_names=["data", "label"])
    first_dataset = first_dataset.map(operations=FirstPyFunc, input_columns=["data"], num_parallel_workers=4,
                                      python_multiprocessing=True)

    class SecondUDFDataset:
        def __init__(self, dataset):
            self.dataset = dataset
            self.dataset_size = self.dataset.get_dataset_size()
            self.iterator = self.dataset.create_dict_iterator(output_numpy=True, num_epochs=1)

        def __next__(self):
            data = next(self.iterator)
            assert data["data"].shape == (300, 300, 3)
            assert data["data"].dtype == np.uint8
            assert data["label"].shape == (1,)
            assert data["label"].dtype == np.uint32
            return data["data"], data["label"]

        def __iter__(self):
            self.iterator = self.dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
            return self

        def __len__(self):
            return self.dataset_size

    def SecondPyFunc(img):
        img = vision.Resize((64, 64))(img)
        img = vision.Rescale(1.0 / 255.0, 0.0)(img)
        img = vision.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])(img)
        return img

    second_dataset = ds.GeneratorDataset(source=SecondUDFDataset(first_dataset), column_names=["data", "label"],
                                         shuffle=False)
    second_dataset = second_dataset.map(operations=SecondPyFunc, input_columns=["data"], num_parallel_workers=2)
    second_dataset = second_dataset.map(operations=transforms.TypeCast(mstype.float32), input_columns=["label"])
    print_psutil("init")

    assert second_dataset.get_dataset_size() == 10
    print_psutil("after dataset_size")
    # TODO: hung with output_shapes & output_types
    ## assert second_dataset.output_shapes() == [[64, 64, 3], [1]]
    ## print_psutil("after shapes")

    ## assert second_dataset.output_types() == [np.float32, np.float32]
    ## print_psutil("after types")

    assert second_dataset.get_col_names() == ["data", "label"]
    print_psutil("after col_names")

    assert second_dataset.get_batch_size() == 1
    print_psutil("batch_size")

    count = 0
    epochs = 3
    epoch = 0
    ds_iter = second_dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    print_psutil("after iterator")
    for _ in range(epochs):
        for item in ds_iter:
            print_psutil("after get item")
            print("epoch: {}, count: {}".format(epoch, count), flush=True)
            assert item['data'].shape == (64, 64, 3)
            assert item['data'].dtype == np.float32
            assert item['label'].dtype == np.float32
            count += 1
        epoch += 1
    assert count == 30
    assert epoch == 3
    print_psutil("end")
    del os.environ["MS_INDEPENDENT_DATASET"]


def test_dataset_with_independent_process_with_dict():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with python dict
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "true"
    diff_shapes = [(548, 507), (778, 577), (1024, 700), (1359, 733), (1570, 882)]
    python_dict = {"filename": "1.jpg", "object": {"truncated": 0, "difficult": 1}, "bndbox": [1, 2, 3, 4]}
    # Random-accessible object as input source
    class RandomAccessDataset:
        def __init__(self):
            self._label = np.zeros((1), dtype=np.float32)
            self.image = np.fromfile(apple_jpg, dtype=np.int8)
            self.sizes = diff_shapes
            self.attr = python_dict

        def __getitem__(self, index):
            img = vision.Decode()(self.image)
            img = vision.Resize(self.sizes[index % 5])(img)
            return self.attr, img, self._label

        def __len__(self):
            return 10

    loader = RandomAccessDataset()
    dataset = ds.GeneratorDataset(source=loader, column_names=["attr", "data", "label"], shuffle=False)
    def add_new_dict(old_dict):
        new_dict = copy.deepcopy(old_dict)
        new_dict["class"] = "cat"
        return old_dict, new_dict
    dataset = dataset.map(operations=add_new_dict, input_columns=["attr"], output_columns=["attr", "attr2"],
                          num_parallel_workers=2, python_multiprocessing=True)

    count = 0
    start = time.time()
    avg = 0
    epochs = 3
    epoch = 0
    shapes_count = [0, 0, 0, 0, 0]
    assert dataset.get_dataset_size() == 10
    assert dataset.output_shapes() == [[0], [0], [548, 507, 3], [1]]
    assert dataset.output_types() == [np.dtype(object), np.dtype(object), np.uint8, np.float32]
    assert dataset.get_col_names() == ["attr", "attr2", "data", "label"]
    assert dataset.get_batch_size() == 1
    new_dict = copy.deepcopy(python_dict)
    new_dict["class"] = "cat"
    ds_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=epochs)
    for _ in range(epochs):
        for item in ds_iter:
            shapes_count[diff_shapes.index(item["data"].shape[0:2])] += 1
            assert isinstance(item["attr"], dict)
            assert item["attr"] == python_dict
            assert isinstance(item["attr2"], dict)
            assert item["attr2"] == new_dict
            if count > 1:
                cost = time.time() - start
                avg += cost
                print("epoch: {}, time cost: {}, count: {}, avg: {}".format(epoch, cost, count, avg / (count - 1)),
                      flush=True)
            count += 1
            start = time.time()
        epoch += 1
    assert len(np.unique(np.array(shapes_count))) == 1
    assert shapes_count[0] == 6
    assert sum(shapes_count) == 30
    assert count == 30
    assert epoch == 3
    del os.environ["MS_INDEPENDENT_DATASET"]


def test_dataset_mnistdataset_with_for_loop_iterator():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with for loop iterator
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "False"

    dataset = ds.MnistDataset(MNIST_DIR)
    dataset = dataset.take(count=5)
    dataset = dataset.skip(count=1)
    dataset = dataset.map(operations=transforms.Unique(), input_columns='image',
                          output_columns=['image', 'image_idx', 'image_cnt'], num_parallel_workers=3,
                          python_multiprocessing=True)
    dataset = dataset.project(columns=['image', 'image_idx', 'image_cnt'])
    dataset = dataset.repeat(5)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(1, drop_remainder=True)
    numiter = 0
    for _ in range(3):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            numiter += 1
    assert numiter == 60

    os.environ["MS_INDEPENDENT_DATASET"] = "True"

    dataset = ds.MnistDataset(MNIST_DIR)
    dataset = dataset.take(count=5)
    dataset = dataset.skip(count=1)
    dataset = dataset.map(operations=transforms.Unique(), input_columns='image',
                          output_columns=['image', 'image_idx', 'image_cnt'], num_parallel_workers=3,
                          python_multiprocessing=True)
    dataset = dataset.project(columns=['image', 'image_idx', 'image_cnt'])
    dataset = dataset.repeat(5)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(1, drop_remainder=True)
    numiter = 0
    for _ in range(3):
        for _ in dataset.create_dict_iterator(output_numpy=True):
            numiter += 1
    assert numiter == 60

    os.environ["MS_INDEPENDENT_DATASET"] = "False"


def test_dataset_minddataset_with_map_error():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with map error
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "True"

    file_name = os.environ.get('PYTEST_CURRENT_TEST').split(':')[-1].split(' ')[0]
    writer = FileWriter(file_name, 4, True)
    data = [{"id": "abc", "label": 1, "rating": 1.1, "input_ids": np.array([1, 2, 3], dtype=np.int64),
             "input_mask": np.array([11, 22, 33], dtype=np.int64),
             "segment_ids": np.array([[11, 11], [22, 22]], dtype=np.int64)},
            {"id": "def", "label": 2, "rating": 2.2, "input_ids": np.array([2, 3, 4], dtype=np.int64),
             "input_mask": np.array([11, 22, 33], dtype=np.int64),
             "segment_ids": np.array([[11, 11], [22, 22]], dtype=np.int64)},
            {"id": "ghi", "label": 3, "rating": 3.3, "input_ids": np.array([3, 4, 5], dtype=np.int64),
             "input_mask": np.array([11, 22, 33], dtype=np.int64),
             "segment_ids": np.array([[11, 11], [22, 22]], dtype=np.int64)},
            {"id": "jkl", "label": 4, "rating": 4.4, "input_ids": np.array([4, 5, 6], dtype=np.int64),
             "input_mask": np.array([11, 22, 33], dtype=np.int64),
             "segment_ids": np.array([[11, 11], [22, 22]], dtype=np.int64)},
            {"id": "mno", "label": 5, "rating": 5.5, "input_ids": np.array([5, 6, 7], dtype=np.int64),
             "input_mask": np.array([11, 22, 33], dtype=np.int64),
             "segment_ids": np.array([[11, 11], [22, 22]], dtype=np.int64)},
            {"id": "pqr", "label": 6, "rating": 6.6, "input_ids": np.array([6, 7, 8], dtype=np.int64),
             "input_mask": np.array([11, 22, 33], dtype=np.int64),
             "segment_ids": np.array([[11, 11], [22, 22]], dtype=np.int64)},
            {"id": "stu", "label": 7, "rating": 7.7, "input_ids": np.array([7, 8, 9], dtype=np.int64),
             "input_mask": np.array([11, 22, 33], dtype=np.int64),
             "segment_ids": np.array([[11, 11], [22, 22]], dtype=np.int64)},
            {"id": "vwx", "label": 8, "rating": 8.8, "input_ids": np.array([8, 9, 10], dtype=np.int64),
             "input_mask": np.array([11, 22, 33], dtype=np.int64),
             "segment_ids": np.array([[11, 11], [22, 22]], dtype=np.int64)}]
    nlp_schema_json = {"id": {"type": "string"},
                       "label": {"type": "int32"},
                       "rating": {"type": "float32"},
                       "input_ids": {"type": "int64", "shape": [-1]},
                       "input_mask": {"type": "int64", "shape": [1, -1]},
                       "segment_ids": {"type": "int64", "shape": [2, -1]}}
    writer.add_schema(nlp_schema_json, "nlp_schema")
    writer.write_raw_data(data)
    writer.commit()

    columns_list = ["input_ids", "input_mask", "segment_ids"]
    num_parallel_workers = 8
    shuffle = True
    num_shards = 4
    shard_id = 1
    dataset = ds.MindDataset(file_name + "0", columns_list, num_parallel_workers, shuffle, num_shards, shard_id)

    def pass_func(_):
        for i in range(10):
            yield (np.array([i]),)

    dataset = dataset.map(operations=pass_func, input_columns=["input_ids"], num_parallel_workers=1)
    num_iter = 0
    with pytest.raises(RuntimeError) as err:
        for _ in dataset.create_dict_iterator(output_numpy=True):
            num_iter += 1
    assert "Exception thrown from dataset pipeline. " in str(err.value) or \
           "Exception thrown from user defined Python" in str(err.value)

    paths = ["{}{}".format(file_name, str(x).rjust(1, '0')) for x in range(4)]
    for x in paths:
        if os.path.exists("{}".format(x)):
            os.remove("{}".format(x))
        if os.path.exists("{}.db".format(x)):
            os.remove("{}.db".format(x))

    os.environ["MS_INDEPENDENT_DATASET"] = "False"


def test_dataset_generator_with_filter_error():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with filter error
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "True"

    def gen(num):
        for i in range(num):
            yield i

    dataset = ds.GeneratorDataset(gen(40), ["num"], num_parallel_workers=8)
    dataset = dataset.repeat(2)

    def apply_func(data):
        data = data.batch(2, num_parallel_workers=6, python_multiprocessing=True)
        return data
    dataset = dataset.apply(apply_func)
    dataset = dataset.filter(predicate=lambda data: data < 11, num_parallel_workers=3)

    def invert_sign_per_batch(collist, batchinfo):
        return ([np.copy(((-1) ** batchinfo.get_batch_num()) * arr) for arr in collist],)
    dataset = dataset.batch(batch_size=2, input_columns=["num"], per_batch_map=invert_sign_per_batch,
                            num_parallel_workers=8, python_multiprocessing=True)
    with pytest.raises(RuntimeError) as err:
        numiter = 0
        for _ in dataset.create_dict_iterator(output_numpy=True):
            numiter += 1
    assert "Exception thrown from dataset pipeline. " in str(err.value) or \
           "Exception thrown from user defined Python function in dataset. " in str(err.value)

    os.environ["MS_INDEPENDENT_DATASET"] = "False"


def test_dataset_wikitextdataset_with_op_input_error():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with invalid op
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "True"

    dataset = ds.WikiTextDataset(WIKI_DIR, usage='train', num_parallel_workers=8)
    unique_op = transforms.Unique()
    dataset = dataset.take(count=5)
    dataset = dataset.skip(count=1)
    dataset = dataset.map(operations=unique_op, input_columns='text', num_parallel_workers=3,
                          python_multiprocessing=False)
    dataset = dataset.project(columns=['text'])
    dataset = dataset.repeat(5)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(32, drop_remainder=True)

    with pytest.raises(RuntimeError) as err:
        count = 0
        for _ in dataset.create_tuple_iterator(num_epochs=1, output_numpy=True):
            count += 1
    assert "Exception thrown from dataset pipeline. " in str(err.value) or \
           "Exception thrown from user defined Python" in str(err.value)

    os.environ["MS_INDEPENDENT_DATASET"] = "False"


def test_dataset_generator_error():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with script error
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "True"

    class RandomAccessDataset:
        def __init__(self):
            self._data = np.ones((5, 2))
            self._label = np.zeros((5, 1))

        def __getitem__(self, index):
            if index == 3:
                return self._data[index], 10 / 0
            return self._data[index], self._label[index]

        def __len__(self):
            return len(self._data)

    # gzj dataset = ds.GeneratorDataset(RandomAccessDataset(), ["num", "label"], num_parallel_workers=8)
    dataset = ds.GeneratorDataset(RandomAccessDataset(), ["num", "label"])

    with pytest.raises(RuntimeError) as err:
        numiter = 0
        for _ in dataset.create_dict_iterator(output_numpy=True):
            numiter += 1
    assert "Exception thrown from dataset pipeline. " in str(err.value) or \
           "Exception thrown from user defined Python function in dataset. " in str(err.value)

    os.environ["MS_INDEPENDENT_DATASET"] = "False"


def test_dataset_GTZANDataset_with_zip_op():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with zip op
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "True"

    dataset = ds.GTZANDataset(dataset_dir=GTZAN_DIR, num_parallel_workers=8, shuffle=True)
    transforms_op = [audio.AllpassBiquad(44100, 200.0)]

    dataset_1 = ds.GTZANDataset(dataset_dir=GTZAN_DIR, num_parallel_workers=8, shuffle=False)
    input_columns = ["waveform", "sample_rate", "label"]
    output_columns = ['a', 'b', 'c']
    dataset_1 = dataset_1.rename(input_columns, output_columns)

    dataset_zip = dataset.zip(dataset_1)
    # operators should be one by one, but got too many branches
    dataset = dataset_zip.map(input_columns=["waveform"], operations=transforms_op,
                              num_parallel_workers=8,
                              python_multiprocessing=True)
    count = 0
    for i in dataset.create_dict_iterator(output_numpy=True):
        assert len(i) == 6
        count += 1
    assert count == 3

    os.environ["MS_INDEPENDENT_DATASET"] = "False"


def test_dataset_TextFile_with_concat_op():
    """
    Feature: Dataset With Independent Process
    Description: Test dataset in independent process with concat op
    Expectation: The dataset is processed as expected
    """
    os.environ["MS_INDEPENDENT_DATASET"] = "True"

    data = ds.TextFileDataset(TEXT_FILE)

    def flat_map_func(x):
        d = ds.MnistDataset(MNIST_DIR)
        return d
    data = data.flat_map(flat_map_func)
    dataset = ds.MnistDataset(MNIST_DIR)
    dataset = dataset.concat(data)
    dataset = dataset.take(count=10)
    dataset = dataset.skip(count=1)
    dataset = dataset.project(columns=['image'])
    dataset = dataset.repeat(5)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.batch(2, drop_remainder=True)
    numiter = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        numiter += 1
    assert numiter == 22

    os.environ["MS_INDEPENDENT_DATASET"] = "False"


if __name__ == "__main__":
    test_dataset_with_independent_process()
    test_dataset_with_independent_process_dynamic_shape()
    test_dataset_with_independent_process_train_and_eval()
    test_dataset_with_independent_process_two_stage_pipeline()
    test_dataset_with_independent_process_with_dict()
    test_dataset_mnistdataset_with_for_loop_iterator()
    test_dataset_minddataset_with_map_error()
    test_dataset_generator_with_filter_error()
    test_dataset_wikitextdataset_with_op_input_error()
    test_dataset_generator_error()
    test_dataset_GTZANDataset_with_zip_op()
    test_dataset_TextFile_with_concat_op()
