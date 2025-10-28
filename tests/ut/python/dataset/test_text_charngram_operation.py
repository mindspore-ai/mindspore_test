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
"""text transform - charngram"""

import os
import numpy as np
import pytest
import mindspore.dataset as ds
from mindspore.dataset import text


TEST_DATA_DATASET_FUNC ="../data/dataset/"

vector_file1 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testVectors", "char_n_gram_20_100d.txt")
vector_file2 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testVectors", "char_n_gram_20_dim_different_100d.txt")
vector_file3 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testVectors", "vector_test.txt")
vector_empty = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testVectors", "vectors_empty.txt")
text_file = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testVectors", "words.txt")


def test_charngram_operation_01():
    """
    Feature: CharNGram op
    Description: Test CharNGram op with different max_vectors and unk_init parameters
    Expectation: Successfully load character n-gram vectors and convert tokens
    """
    # Description: test with only default parameter
    char_n_gram = text.CharNGram.from_file(vector_file1)
    a = []
    for _ in range(100):
        a.append(10)
    vectors_op = text.ToVectors(char_n_gram, a)
    data = ds.TextFileDataset(text_file, shuffle=False)
    data = data.map(operations=vectors_op, input_columns=["text"])
    ind = 0
    out = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        out.append(d["text"])
        ind += 1
    assert ind == 7
    assert (out[0] == a).all()

    # Description: test with only default parameter
    char_n_gram = text.CharNGram.from_file(vector_file1, 300)
    a = []
    for _ in range(100):
        a.append(10)
    vectors_op = text.ToVectors(char_n_gram, a)
    data = ds.TextFileDataset(text_file, shuffle=False)
    data = data.map(operations=vectors_op, input_columns=["text"])
    ind = 0
    out = []
    for d in data.create_dict_iterator(num_epochs=1, output_numpy=True):
        out.append(d["text"])
        ind += 1
    assert ind == 7
    assert (out[0] == a).all()

    # Description: test with all parameters which include `path` and `max_vector` in function BuildFromFile in eager mode
    char_n_gram = text.CharNGram.from_file(vector_file1, max_vectors=18)
    vectors_op = text.ToVectors(char_n_gram)
    vectors_op("the")

    # test apply_func return ""
    def generator():
        text_list = ["te", "Ba", "ab", "D", "haha", "!", "%^", "1", "张"]
        for i in text_list:
            yield (np.array([i]),)

    char_n_gram = text.CharNGram.from_file(vector_file1, 10)
    a = []
    for i in range(100):
        a.append(i)
    vectors_op = text.ToVectors(char_n_gram, a)
    dataset = ds.GeneratorDataset(generator, ["text"], shuffle=False)
    dataset = dataset.map(operations=vectors_op, input_columns=["text"])
    numiter = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        numiter += 1
    assert numiter == 9

    # test apply_func return ""
    def generator2():
        text_list = ["@#$", "l", " ", "//", "张"]
        for i in text_list:
            yield (np.array([i]),)

    char_n_gram = text.CharNGram.from_file(vector_file1, 100)
    a = []
    for i in range(100):
        a.append(i)
    vectors_op = text.ToVectors(char_n_gram, a)
    dataset = ds.GeneratorDataset(generator2, ["text"], shuffle=False)
    dataset = dataset.map(operations=vectors_op, input_columns=["text"])
    numiter = 0
    for _ in dataset.create_dict_iterator(output_numpy=True):
        numiter += 1
    assert numiter == 5


def test_charngram_exception_01():
    """
    Feature: CharNGram op
    Description: Test CharNGram op with invalid file formats and parameters
    Expectation: Raise expected exceptions for invalid vector files and parameters
    """
    # Description: test with only default parameter
    with pytest.raises(RuntimeError, match="Vectors: all vectors must have the same number"
                                           " of dimensions, but got dim 99 while expecting 100"):
        text.CharNGram.from_file(vector_file2)

    # vector is empty
    with pytest.raises(RuntimeError, match="Vectors: invalid file, file is empty."):
        text.CharNGram.from_file(vector_empty)

    # vector is empty
    with pytest.raises(TypeError, match="Argument file_path with value 1 is not of"
                                        " type \\[<class 'str'>\\], but got <class 'int'>."):
        text.CharNGram.from_file(1)

    # vector is empty
    with pytest.raises(RuntimeError, match="Vectors: "):
        text.CharNGram.from_file("no_text")

    # Description: test with only default parameter
    with pytest.raises(RuntimeError, match="ToVectors: unk_init must be the same length as"
                                           " vectors, but got unk_init: 10 and vectors: 100"):
        char_n_gram = text.CharNGram.from_file(vector_file1)
        a = []
        for _ in range(10):
            a.append(10)
        vectors_op = text.ToVectors(char_n_gram, a)
        data = ds.TextFileDataset(text_file, shuffle=False)
        data = data.map(operations=vectors_op, input_columns=["text"])
        for _ in data.create_dict_iterator(num_epochs=1, output_numpy=True):
            pass

    # vector is empty
    with pytest.raises(ValueError,
                       match="Input max_vectors is not within the required interval of \\[0, 2147483647\\]."):
        text.CharNGram.from_file(vector_file1, -1)

    # vector is empty
    with pytest.raises(ValueError,
                       match="Input max_vectors is not within the required interval of \\[0, 2147483647\\]."):
        text.CharNGram.from_file(vector_file1, 2147483648)

    # vector is empty
    with pytest.raises(TypeError, match="Argument max_vectors with value 10.0 is not of"
                                        " type \\[<class 'int'>\\], but got <class 'float'>."):
        text.CharNGram.from_file(vector_file1, 10.0)

    # vector is empty
    with pytest.raises(TypeError, match="Argument max_vectors with value True is not of"
                                        " type \\(<class 'int'>,\\), but got <class 'bool'>."):
        text.CharNGram.from_file(vector_file1, True)

    # vector is empty
    with pytest.raises(TypeError, match="missing a required argument: 'file_path'"):
        text.CharNGram.from_file()
