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
"""text transform - wordpiecetokenizer"""

import os
import numpy as np
import pytest
import mindspore.dataset as ds
import mindspore.dataset.text as nlp


TEST_DATA_DATASET_FUNC ="../data/dataset/"

DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC,
                         "text_data/testTextFile/textfile/testTokenizerData/WordpieceTokenizer/english.txt")
DATA_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/WordpieceTokenizer/chinese.txt")
DATA_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC,
                          "text_data/testTextFile/textfile/testTokenizerData/WordpieceTokenizer/mixed.txt")

vocab_english = ["book", "cholera", "era", "favor", "##ite", "my", "is", "love", "dur", "##ing", "the"]

vocab_chinese = ["我", '最', '喜', '欢', '的', '书', '是', '霍', '乱', '时', '期', '爱', '情']

vocab_mix = vocab_chinese + vocab_english


def test_wordpiecetokenizer_operation_01():
    """
    Feature: Test wordpiecetokenizer
    Description: Test parameters vocab, max_bytes_per_token
    Expectation: success
    """

    # test vocab of english and unknown_token=''
    vocab_list = ["book", "cholera", "era", "favor", "##ite", "my", "is", "love", "dur", "##ing", "the"]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    vocab = nlp.Vocab.from_list(vocab_list)
    tokenizer_op = nlp.WordpieceTokenizer(vocab=vocab, unknown_token='')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    expect_str = [['my'], ['favor', '##ite'], ['book'], ['is'], ['love'], ['dur', '##ing'], ['the'], ['cholera'],
                  ['era'], ['what']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text']
        np.testing.assert_array_equal(text, expect_str[count])
        count = count + 1

    # test vocab of chinese and unknown_token=''
    vocab_list = ["我", '最', '喜', '欢', '的', '书', '是', '霍', '乱', '时', '期', '爱', '情']
    dataset = ds.TextFileDataset(DATA_FILE1, shuffle=False)
    vocab = nlp.Vocab.from_list(vocab_list)
    tokenizer_op = nlp.WordpieceTokenizer(vocab=vocab, unknown_token='')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    expect_str = [['我'], ['最'], ['喜'], ['欢'], ['的'], ['书'], ['是'], ['霍'], ['乱'], ['时'], ['期'], ['的'],
                  ['爱'], ['情'], ['您']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text']
        np.testing.assert_array_equal(text, expect_str[count])
        count = count + 1

    # test vocab of english and chinese, unknown_token='[UNK]'
    vocab_list = ["我", '最', '喜', '欢', '的', '书', '是', '霍', '乱', '时', '期', '爱', '情', "book", "cholera", "era",
                  "favor", "##ite", "my", "is", "love", "dur", "##ing", "the"]
    dataset = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    vocab = nlp.Vocab.from_list(vocab_list)
    tokenizer_op = nlp.WordpieceTokenizer(vocab=vocab, unknown_token='[UNK]')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    expect_str = [['my'], ['favor', '##ite'], ['book'], ['is'], ['love'], ['dur', '##ing'], ['the'], ['cholera'],
                  ['era'], ['[UNK]'], ['我'], ['最'], ['喜'], ['欢'], ['的'], ['书'], ['是'], ['霍'], ['乱'],
                  ['时'], ['期'], ['的'], ['爱'], ['情'], ['[UNK]']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text']
        np.testing.assert_array_equal(text, expect_str[count])
        count = count + 1

    # test vocab of english and chinese, unknown_token=''
    vocab_list = ["我", '最', '喜', '欢', '的', '书', '是', '霍', '乱', '时', '期', '爱', '情', "book", "cholera", "era", "favor",
                  "##ite", "my", "is", "love", "dur", "##ing", "the"]
    dataset = ds.TextFileDataset(DATA_FILE2, shuffle=False)
    vocab = nlp.Vocab.from_list(vocab_list)
    tokenizer_op = nlp.WordpieceTokenizer(vocab=vocab, unknown_token='')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    expect_str = [['my'], ['favor', '##ite'], ['book'], ['is'], ['love'], ['dur', '##ing'], ['the'], ['cholera'],
                  ['era'], ['what'], ['我'], ['最'], ['喜'], ['欢'], ['的'], ['书'], ['是'], ['霍'], ['乱'],
                  ['时'], ['期'], ['的'], ['爱'], ['情'], ['您']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text']
        np.testing.assert_array_equal(text, expect_str[count])
        count = count + 1

    # test max_bytes_per_token = 7
    vocab_list = ["book", "cholera", "era", "favor", "##ite", "my", "is", "love", "dur", "##ing", "the"]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    vocab = nlp.Vocab.from_list(vocab_list)
    tokenizer_op = nlp.WordpieceTokenizer(vocab=vocab, max_bytes_per_token=7, unknown_token='[UNK]')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    expect_str = [['my'], ['[UNK]'], ['book'], ['is'], ['love'], ['dur', '##ing'], ['the'], ['cholera'], ['era'],
                  ['[UNK]']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text']
        np.testing.assert_array_equal(text, expect_str[count])
        count = count + 1

    # test max_bytes_per_token = 8
    vocab_list = ["book", "cholera", "era", "favor", "##ite", "my", "is", "love", "dur", "##ing", "the"]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    vocab = nlp.Vocab.from_list(vocab_list)
    tokenizer_op = nlp.WordpieceTokenizer(vocab=vocab, max_bytes_per_token=8, unknown_token='[UNK]')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    expect_str = [['my'], ['favor', '##ite'], ['book'], ['is'], ['love'], ['dur', '##ing'], ['the'], ['cholera'],
                  ['era'], ['[UNK]']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text']
        np.testing.assert_array_equal(text, expect_str[count])
        count = count + 1


def test_wordpiecetokenizer_operation_02():
    """
    Feature: Test wordpiecetokenizer
    Description: Test parameters max_bytes_per_token, suffix_indicator, unknown_token, …
    Expectation: success
    """

    # test max_bytes_per_token = 0
    vocab_list = ["book", "cholera", "era", "favor", "##ite", "my", "is", "love", "dur", "##ing", "the"]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    vocab = nlp.Vocab.from_list(vocab_list)
    tokenizer_op = nlp.WordpieceTokenizer(vocab=vocab, max_bytes_per_token=0, unknown_token='[UNK]')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    expect_str = [['[UNK]'], ['[UNK]'], ['[UNK]'], ['[UNK]'], ['[UNK]'], ['[UNK]'], ['[UNK]'], ['[UNK]'], ['[UNK]'],
                  ['[UNK]']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text']
        np.testing.assert_array_equal(text, expect_str[count])
        count = count + 1

    # test suffix_indicator = '**'
    vocab_list = ["book", "cholera", "era", "favor", "**ite", "my", "is", "love", "dur", "**ing", "the"]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    vocab = nlp.Vocab.from_list(vocab_list)
    tokenizer_op = nlp.WordpieceTokenizer(vocab=vocab, suffix_indicator='**', unknown_token='[UNK]')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    expect_str = [['my'], ['favor', '**ite'], ['book'], ['is'], ['love'], ['dur', '**ing'], ['the'], ['cholera'],
                  ['era'], ['[UNK]']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text']
        np.testing.assert_array_equal(text, expect_str[count])
        count = count + 1

    # test unknown_token='ascd9&*^'
    vocab_list = ["book", "cholera", "era", "favor", "##ite", "my", "is", "love", "dur", "##ing", "the"]
    dataset = ds.TextFileDataset(DATA_FILE, shuffle=False)
    vocab = nlp.Vocab.from_list(vocab_list)
    tokenizer_op = nlp.WordpieceTokenizer(vocab=vocab, unknown_token='ascd9&*^')
    dataset = dataset.map(operations=tokenizer_op)
    count = 0
    expect_str = [['my'], ['favor', '##ite'], ['book'], ['is'], ['love'], ['dur', '##ing'], ['the'], ['cholera'],
                  ['era'], ['ascd9&*^']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        text = i['text']
        np.testing.assert_array_equal(text, expect_str[count])
        count = count + 1

    # test vocab of chinese and with_offsets = True
    vocab_list = ["我", '最', '喜', '欢', '的', '书', '是', '霍', '乱', '时', '期', '爱', '情']
    dataset = ds.TextFileDataset(DATA_FILE1, shuffle=False)
    vocab = nlp.Vocab.from_list(vocab_list)
    tokenizer_op = nlp.WordpieceTokenizer(vocab=vocab, unknown_token='[UNK]', with_offsets=True)
    dataset = dataset.map(input_columns=["text"], output_columns=["token", "offsets_start", "offsets_limit"],
                          operations=tokenizer_op)
    dataset = dataset.project(columns=["token", "offsets_start", "offsets_limit"])
    count = 0
    expected_offsets_start = [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]]
    expected_offsets_limit = [[3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3], [3]]
    expect_str = [['我'], ['最'], ['喜'], ['欢'], ['的'], ['书'], ['是'], ['霍'], ['乱'], ['时'], ['期'], ['的'], ['爱'], ['情'],
                  ['[UNK]']]
    for i in dataset.create_dict_iterator(output_numpy=True):
        token = i['token']
        np.testing.assert_array_equal(token, expect_str[count])
        np.testing.assert_array_equal(i['offsets_start'], expected_offsets_start[count])
        np.testing.assert_array_equal(i['offsets_limit'], expected_offsets_limit[count])
        count = count + 1


def test_wordpiecetokenizer_exception_01():
    """
    Feature: Test wordpiecetokenizer
    Description: Test parameters suffix_indicator with exception
    Expectation: success
    """
    # test suffix_indicator = 2
    vocab_list = ["book", "cholera", "era", "favor", "**ite", "my", "is", "love", "dur", "**ing", "the"]
    vocab = nlp.Vocab.from_list(vocab_list)
    with pytest.raises(TypeError, match="Wrong input type for suffix_indicator, should be string"):
        nlp.WordpieceTokenizer(vocab=vocab, suffix_indicator=2, unknown_token='[UNK]')
