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
"""text transform - fasttext"""

import os
import pytest
import numpy as np
import mindspore.dataset as ds
from mindspore.dataset import text
import mindspore.dataset.text.transforms as t_trans


TEST_DATA_DATASET_FUNC ="../data/dataset/"


def test_fasttext_operation_01():
    """
    Feature: FastText op
    Description: Test FastText op with default and all parameters, including eager mode
    Expectation: Successfully generate word vectors and match expected results
    """
    # FastText:Test with only default parameter
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    vectors = text.FastText.from_file(dataset_dir + "fast_text.vec")
    to_vectors = text.ToVectors(vectors)
    data = ds.TextFileDataset(dataset_dir + "words.txt", shuffle=False)
    data = data.map(operations=to_vectors, input_columns=["text"])
    ind = 0
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0, 0, 0, 0, 0, 0],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923],
           [0, 0, 0, 0, 0, 0]]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_array = np.array(res[ind], dtype=np.float32)
        assert np.array_equal(res_array, d["text"]), ind
        ind += 1

    # FastText:Test with all parameters in function BuildFromFile
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    vectors = text.FastText.from_file(dataset_dir + "fast_text.vec", max_vectors=100)
    to_vectors = text.ToVectors(vectors)
    data = ds.TextFileDataset(dataset_dir + "words.txt", shuffle=False)
    data = data.map(operations=to_vectors, input_columns=["text"])
    ind = 0
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0, 0, 0, 0, 0, 0],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923],
           [0, 0, 0, 0, 0, 0]]
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        res_array = np.array(res[ind], dtype=np.float32)
        assert np.array_equal(res_array, d["text"]), ind
        ind += 1

    # FastText:Test with all parameters in eager mode
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    vectors = text.FastText.from_file(dataset_dir + "fast_text.vec", max_vectors=4)
    to_vectors = t_trans.ToVectors(vectors)
    result1 = to_vectors("ok")
    result2 = to_vectors("!")
    result3 = to_vectors("this")
    result4 = to_vectors("is")
    result5 = to_vectors("my")
    result6 = to_vectors("home")
    result7 = to_vectors("none")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0]]
    res_array = np.array(res, dtype=np.float32)

    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])

    # FastText:Test with all parameters which include `unk_init` and `lower_case_backup` in function ToVectors
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    vectors = text.FastText.from_file(dataset_dir + "fast_text.vec", max_vectors=4)
    my_unk = [-1, -1, -1, -1, -1, -1]
    to_vectors = t_trans.ToVectors(vectors, unk_init=my_unk, lower_case_backup=True)
    result1 = to_vectors("Ok")
    result2 = to_vectors("!")
    result3 = to_vectors("This")
    result4 = to_vectors("is")
    result5 = to_vectors("my")
    result6 = to_vectors("home")
    result7 = to_vectors("none")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [-1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1],
           [-1, -1, -1, -1, -1, -1]]
    res_array = np.array(res, dtype=np.float32)

    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])


def test_fasttext_operation_02():
    """
    Feature: FastText op
    Description: Test FastText op in eager mode with default parameters
    Expectation: Successfully convert words to vectors in eager mode
    """
    # FastText: test with only default parameter in eager mode
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    vectors = text.FastText.from_file(dataset_dir + "fast_text.vec")
    to_vectors = t_trans.ToVectors(vectors)
    result1 = to_vectors("ok")
    result2 = to_vectors("!")
    result3 = to_vectors("this")
    result4 = to_vectors("is")
    result5 = to_vectors("my")
    result6 = to_vectors("home")
    result7 = to_vectors("none")
    res = [[0.418, 0.24968, -0.41242, 0.1217, 0.34527, -0.04445718411],
           [0.013441, 0.23682, -0.16899, 0.40951, 0.63812, 0.47709],
           [0.15164, 0.30177, -0.16763, 0.17684, 0.31719, 0.33973],
           [0.70853, 0.57088, -0.4716, 0.18048, 0.54449, 0.72603],
           [0.68047, -0.039263, 0.30186, -0.17792, 0.42962, 0.032246],
           [0.26818, 0.14346, -0.27877, 0.016257, 0.11384, 0.69923],
           [0, 0, 0, 0, 0, 0]]
    res_array = np.array(res, dtype=np.float32)

    assert np.array_equal(result1, res_array[0])
    assert np.array_equal(result2, res_array[1])
    assert np.array_equal(result3, res_array[2])
    assert np.array_equal(result4, res_array[3])
    assert np.array_equal(result5, res_array[4])
    assert np.array_equal(result6, res_array[5])
    assert np.array_equal(result7, res_array[6])


def test_fasttext_exception_01():
    """
    Feature: FastText op
    Description: Test FastText op with invalid parameters and file formats
    Expectation: Raise expected exceptions for invalid inputs
    """
    # FastText:Test not all vectors have the same number of dimensions
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    with pytest.raises(RuntimeError, match="all vectors must have the same number of dimensions, "
                                           "but got dim 5 while expecting 6"):
        vectors = text.FastText.from_file(dataset_dir + "fast_text_dim_different.vec")
        to_vectors = t_trans.ToVectors(vectors)
        to_vectors("ok")

    # FastText:Test the file is empty.
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    with pytest.raises(RuntimeError, match="invalid file, file is empty."):
        vectors = text.FastText.from_file(dataset_dir + "fast_text_empty.vec")
        to_vectors = t_trans.ToVectors(vectors)
        to_vectors("ok")

    # FastText:Test the file not exist
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    with pytest.raises(RuntimeError, match="FastText: invalid file"):
        vectors = text.FastText.from_file(dataset_dir + "not_exist.vec")
        to_vectors = t_trans.ToVectors(vectors)
        to_vectors("ok")

    # FastText:Test the token is 1-dimensional
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    with pytest.raises(RuntimeError, match="token with 1-dimensional vector"):
        vectors = text.FastText.from_file(dataset_dir + "fast_text_with_wrong_info.vec")
        to_vectors = t_trans.ToVectors(vectors)
        to_vectors("ok")

    # FastText:Test max_vectors parameter < 0
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    with pytest.raises(ValueError, match="Input max_vectors is not within the required interval"):
        vectors = text.FastText.from_file(dataset_dir + "fast_text.vec", max_vectors=-1)
        to_vectors = t_trans.ToVectors(vectors)
        to_vectors("ok")

    # FastText:Test max_vectors parameter type is a float
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    with pytest.raises(TypeError, match="Argument max_vectors with value 1.0 is not of type \\[<class 'int'>\\], "
                                        "but got <class 'float'>"):
        vectors = text.FastText.from_file(dataset_dir + "fast_text.vec", max_vectors=1.0)
        to_vectors = t_trans.ToVectors(vectors)
        to_vectors("ok")

    # FastText:Test max_vectors parameter type is a string
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    with pytest.raises(TypeError, match="Argument max_vectors with value 1 is not of type \\[<class 'int'>\\],"
                                        " but got <class 'str'>."):
        vectors = text.FastText.from_file(dataset_dir + "fast_text.vec", max_vectors="1")
        to_vectors = t_trans.ToVectors(vectors)
        to_vectors("ok")

    # FastText:Test the suffix of pre-training is not `*.vec'
    dataset_dir = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testFastText", "")
    with pytest.raises(RuntimeError,
                       match="FastText: invalid file, can not find file '\\*.vec', but got:"):
        vectors = text.FastText.from_file(dataset_dir + "fast_text.txt")
        to_vectors = t_trans.ToVectors(vectors)
        to_vectors("ok")
