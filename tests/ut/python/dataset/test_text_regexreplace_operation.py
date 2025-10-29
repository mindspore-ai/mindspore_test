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
"""text transform - regexreplace"""

import os
import pytest
import mindspore.dataset as ds
import mindspore.dataset.text as nlp


TEST_DATA_DATASET_FUNC ="../data/dataset/"


DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data/testTextFile/textfile/testTokenizerData/RegexReplace/1.txt")
DATA_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/RegexReplace/2.txt")
DATA_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/RegexReplace/3.txt")
DATA_FILE3 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/RegexReplace/4.txt")


def test_regexreplace_operation_01():
    """
    Feature: RegexReplace op
    Description: Test RegexReplace op with different patterns and replace_all settings
    Expectation: Successfully replace matched patterns in strings
    """
    # Test RegexReplace,"^(\\d:|b:)"
    pattern = "^(\\d:|b:)"
    replace = ""
    dataset = ds.TextFileDataset(DATA_FILE1, shuffle=False)
    replace_op = nlp.RegexReplace(pattern=pattern, replace=replace)
    dataset = dataset.map(operations=replace_op)
    out_text = []
    expect_str = ['hello', 'world', '31:beijing']
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text'].tolist()
        out_text.append(text)
    assert expect_str == out_text

    # Test RegexReplace,"\\s+"
    pattern = "\\s+"
    replace = ""
    dataset = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    replace_op = nlp.RegexReplace(pattern=pattern, replace=replace)
    dataset = dataset.map(operations=replace_op)
    out_text = []
    expect_str = ["WelcometoChina!"]
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text'].tolist()
        out_text.append(text)
    assert expect_str == out_text

    # Test RegexReplace,replace_all=True
    pattern = "one"
    replace = "two"
    data = 'onetwoonetwoone'
    replace_op = nlp.RegexReplace(pattern=pattern, replace=replace, replace_all=True)
    result = replace_op(data)
    assert result == 'twotwotwotwotwo'

    # Test RegexReplace,replace_all=False
    pattern = "one"
    replace = "two"
    data = 'onetwoonetwoone'
    replace_op = nlp.RegexReplace(pattern=pattern, replace=replace, replace_all=False)
    result = replace_op(data)
    assert result == 'twotwoonetwoone'


def test_regexreplace_exception_01():
    """
    Feature: RegexReplace op
    Description: Test RegexReplace op with invalid parameter types
    Expectation: Raise expected exceptions for invalid parameter types
    """
    # Test RegexReplace,replace_all=0
    pattern = "one"
    replace = "two"
    with pytest.raises(TypeError, match=r'Argument replace_all with value 0 is not of type \[\<class \'bool\'\>\].'):
        _ = nlp.RegexReplace(pattern=pattern, replace=replace, replace_all=0)

    # Test RegexReplace,pattern=True
    replace = "two"
    with pytest.raises(TypeError, match=r'Argument pattern with value True is not of type \[\<class \'str\'\>\].'):
        _ = nlp.RegexReplace(pattern=True, replace=replace, replace_all=True)

    # Test RegexReplace,replace=False
    pattern = "one"
    with pytest.raises(TypeError, match=r'Argument replace with value False is not of type \[\<class \'str\'\>\].'):
        _ = nlp.RegexReplace(pattern=pattern, replace=False, replace_all=True)
