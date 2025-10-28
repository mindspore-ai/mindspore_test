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
"""text transform - pythontokenizer"""

import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.text as nlp


TEST_DATA_DATASET_FUNC ="../data/dataset/"


DATA_FILE = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testJiebaDataset/file1.txt"
DATA_FILE2 = TEST_DATA_DATASET_FUNC + "/text_data/testTextFile/textfile/testJiebaDataset/file2.txt"


def gen_mix():
    text = "with 007 中文 ¥<&gt;& @*test#"
    yield (text,)


def pytoken_op(input_data):
    te = input_data
    tokens = []
    tokens.append(te[:2])
    tokens.append(te[2:7])
    tokens.append(te[10:16])
    tokens.append(te[16:])
    return list(tokens)


def pytoken_op2(input_data):
    return input_data.split()


def pytoken_op3():
    tokens = ['Hello', 'welcome!']
    return tokens


def pytoken_op4(_):
    pass


def pytoken_op5(_):
    return 1


def pytoken_op6(_):
    return None


def pytoken_op7(_):
    return ''


def test_pythontokenizer_operation_01():
    """
    Feature: PythonTokenizer op
    Description: Test PythonTokenizer op with custom Python functions
    Expectation: Successfully tokenize strings using Python functions
    """
    # tokenizer = function,input = english
    data = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    tokenizer = nlp.PythonTokenizer(pytoken_op2)
    dataset = data.map(operations=tokenizer, num_parallel_workers=1)
    expect = ['Hello', 'welcome', 'to', 'the', 'hotline', 'of', 'JinTaiLong,', 'we', 'will', 'do', 'our', 'best', 'to',
              'server', 'you!']
    out = []
    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        s = i['text'].tolist()
        out.append(s)
    np.testing.assert_array_equal(expect, out[0])

    # tokenizer = function,input = english
    data = 'Hello yes'
    tokenizer = nlp.PythonTokenizer(pytoken_op2)
    with pytest.raises(TypeError, match=r'input should be a NumPy array. Got \<class \'str\'\>.'):
        _ = tokenizer(data)

    # tokenizer = function,input = english
    data = np.array('Hello world'.encode())
    tokenizer = nlp.PythonTokenizer(pytoken_op2)
    res = tokenizer(data)
    assert np.array_equal(res, ['Hello', 'world'])

    # tokenizer = function,input = english
    data = ['Hello yes', 'world']
    tokenizer = nlp.PythonTokenizer(pytoken_op2)
    with pytest.raises(TypeError, match=r'input should be a NumPy array. Got \<class \'list\'\>.'):
        _ = tokenizer(data)

    # tokenizer = function,input = english
    data = np.array('Hello world'.encode())
    tokenizer = nlp.PythonTokenizer(pytoken_op7)
    with pytest.raises(RuntimeError, match=r"Pyfunc \[pytoken_op7\], error message: 0-dimensional argument does "
                                           r"not have enough dimensions for all core dimensions \(\'n\',\)"):
        _ = tokenizer(data)


def test_pythontokenizer_exception_01():
    """
    Feature: PythonTokenizer op
    Description: Test PythonTokenizer op with invalid functions and parameters
    Expectation: Raise expected exceptions for invalid inputs
    """
    # tokenizer = function
    data = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    with pytest.raises(RuntimeError):
        tokenizer = nlp.PythonTokenizer(pytoken_op3)
        dataset = data.map(operations=tokenizer, num_parallel_workers=1)
        for i in dataset.create_dict_iterator(output_numpy=True):
            i['text'].tolist()

    # tokenizer = function
    data = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    with pytest.raises(RuntimeError):
        tokenizer = nlp.PythonTokenizer(pytoken_op4)
        dataset = data.map(operations=tokenizer, num_parallel_workers=1)
        for i in dataset.create_dict_iterator(output_numpy=True):
            i['text'].tolist()

    # tokenizer = function
    data = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    with pytest.raises(RuntimeError):
        tokenizer = nlp.PythonTokenizer(pytoken_op5)
        dataset = data.map(operations=tokenizer, num_parallel_workers=1)
        for i in dataset.create_dict_iterator(output_numpy=True):
            i['text'].tolist()

    # tokenizer = function
    data = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    with pytest.raises(RuntimeError):
        tokenizer = nlp.PythonTokenizer(pytoken_op6)
        dataset = data.map(operations=tokenizer, num_parallel_workers=1)
        for i in dataset.create_dict_iterator(output_numpy=True):
            i['text'].tolist()

    # tokenizer = number
    data = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    with pytest.raises(TypeError):
        tokenizer = nlp.PythonTokenizer("test")
        dataset = data.map(operations=tokenizer, num_parallel_workers=1)
        for i in dataset.create_dict_iterator(output_numpy=True):
            i['text'].tolist()
