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
"""text transform - jiebatokenizer"""

import os
import platform
import pytest
import mindspore.dataset as ds
from mindspore.dataset.text import JiebaTokenizer
from mindspore.dataset.text import JiebaMode
import mindspore.dataset.text as nlp


TEST_DATA_DATASET_FUNC ="../data/dataset/"


JIEBATOKENIZER_FILE1 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                    "testJiebaDataset", "file1.txt")
JIEBATOKENIZER_FILE2 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                    "testJiebaDataset", "file2.txt")
JIEBATOKENIZER_FILE3 = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                    "testJiebaDataset", "file3.txt")
JIEBATOKENIZER_HMM_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                       "jiebadict", "hmm_model.utf8")
JIEBATOKENIZER_MP_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                      "jiebadict", "jieba.dict.utf8")


def test_jiebatokenizer_operation_01():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with different modes and add_word/add_dict functions
    Expectation: Successfully tokenize Chinese text with custom dictionaries
    """
    # Test jieba tokenizer with no mode
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE1)
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE)
    data = data.map(operations=jieba_op,
                    input_columns=["text"], num_parallel_workers=1)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test jieba tokenizer with english
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE2)
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    expect = ['Hello', ' ', 'welcome', ' ', 'to', ' ', 'the', ' ', 'hotline', ' ', 'of',
              ' ', 'JinTaiLong', ',', ' ', 'we', ' ', 'will', ' ', 'do', ' ', 'our', ' ',
              'best', ' ', 'to', ' ', 'server', ' ', 'you', '!']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test jieba tokenizer with add word,freq is 0
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_word('北京清华大学', freq=0)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test jieba tokenizer with add word,freq is 15
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_word('北京清华大学', freq=150)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test jieba tokenizer with add word,freq is 500090
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_word('北京清华大学', freq=500090)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=2)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test add_dict with dict
    user_dict = {
        "北京清华大学": 10
    }
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_dict(user_dict)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test add_dict with dict
    user_dict = {
        "北京清华大学": 10,
        "硕士毕业": 10000
    }
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_dict(user_dict)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['小明', '硕士毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]


def test_jiebatokenizer_operation_02():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with dict files and with_offsets parameter
    Expectation: Successfully load user dictionaries and provide offset information
    """
    # Test add_dict with empty dict
    user_dict = {}
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_dict(user_dict)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['小明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '北京', '清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test add_dict with empty dict
    if platform.system() == "Windows":
        jiebatokenizer_file = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                           "testJiebaDataset", "user_dict_win.txt")
    else:
        jiebatokenizer_file = os.path.join(TEST_DATA_DATASET_FUNC, "text_data", "testTextFile", "textfile",
                                           "testJiebaDataset", "user_dict.txt")
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE3)
    jieba_op = JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    jieba_op.add_dict(jiebatokenizer_file)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    num_parallel_workers=1)
    expect = ['小明', '硕士毕业', '于', '中国科学院', '计算所', '，', '北京清华大学', '后', '在', '日本京都大学', '深造']
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["text"]
        for index, item in enumerate(ret):
            assert item == expect[index]

    # Test with_offsets is True
    data = ds.TextFileDataset(JIEBATOKENIZER_FILE1)
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MP, with_offsets=True)
    data = data.map(operations=jieba_op, input_columns=["text"],
                    output_columns=["token", "offsets_start", "offsets_limit"], num_parallel_workers=1)
    data = data.project(columns=["token", "offsets_start", "offsets_limit"])
    expect = ['小', '明', '硕士', '毕业', '于', '中国科学院', '计算所', '，', '后', '在', '日本京都大学', '深造']
    expected_offsets_start = [0, 3, 6, 12, 18, 21, 36, 45, 48, 51, 54, 72]
    expected_offsets_limit = [3, 6, 12, 18, 21, 36, 45, 48, 51, 54, 72, 78]
    for i in data.create_dict_iterator(output_numpy=True):
        ret = i["token"]
        for index, item in enumerate(ret):
            assert item == expect[index]
        for index, item in enumerate(i["offsets_start"]):
            assert item == expected_offsets_start[index]
        for index, item in enumerate(i["offsets_limit"]):
            assert item == expected_offsets_limit[index]

    # Test jieba tokenizer default value
    data = "我爱我的家乡"
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE)
    data1 = jieba_op(data)

    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX,
                                  with_offsets=False)
    data2 = jieba_op(data)
    assert (data1 == data2).all()

    # Test jieba tokenizer mode is JiebaMode.MP
    data = "我爱我的家乡"
    data1 = ['我', '爱', '我', '的', '家乡']
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MP)
    data = jieba_op(data)
    assert (data == data1).all()

    # Test jieba tokenizer mode is JiebaMode.HMM
    data = "我爱我的家乡"
    data1 = ['我', '爱', '我', '的', '家', '乡']
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    data = jieba_op(data)
    assert (data == data1).all()

    # Test jieba tokenizer mode is JiebaMode.HMM
    data = "home 123"
    data1 = ['home', ' ', '123']
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    data = jieba_op(data)
    assert (data == data1).all()

    # Test jieba tokenizer mode is JiebaMode.HMM
    data = "申猴酉鸡@哈喽"
    data1 = ['申猴', '酉', '鸡', '@', '哈', '喽']
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    data = jieba_op(data)
    assert (data == data1).all()

    # Test jieba tokenizer mode is JiebaMode.HMM
    data = "今天是元宵节，猜灯谜！"
    data1 = ['今天', '是', '元宵节', '，', '猜灯谜', '！']
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    data = jieba_op(data)
    assert (data == data1).all()


def test_jiebatokenizer_exception_01():
    """
    Feature: JiebaTokenizer op
    Description: Test JiebaTokenizer op with invalid parameters and input types
    Expectation: Raise expected exceptions for invalid inputs
    """
    # Test jieba tokenizer with error mode
    with pytest.raises(TypeError, match='Wrong input type for mode, should be JiebaMode'):
        nlp.JiebaTokenizer(hmm_path=JIEBATOKENIZER_HMM_FILE, mp_path=JIEBATOKENIZER_MP_FILE, mode='HIM')

    # Test jieba tokenizer with no mp_path
    with pytest.raises(TypeError, match="missing a required argument: 'mp_path'"):
        nlp.JiebaTokenizer(hmm_path=JIEBATOKENIZER_HMM_FILE, mode=JiebaMode.MIX)

    # Test jieba tokenizer with no hmm_path
    with pytest.raises(TypeError, match="missing a required argument: 'hmm_path'"):
        nlp.JiebaTokenizer(mp_path=JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)

    # Test jieba tokenizer with no hmm_path and mp_path
    with pytest.raises(TypeError, match="missing a required argument: 'hmm_path'"):
        nlp.JiebaTokenizer(mode=JiebaMode.MIX)

    # Test jieba tokenizer with add word,freq is -1
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    with pytest.raises(ValueError, match=r"Input is not within the required interval of \[0, 4294967295\]"):
        jieba_op.add_word('北京清华大学', freq=-1)

    # Test jieba tokenizer with add word,freq is string
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    with pytest.raises(TypeError, match="Argument  with value test is not of type"):
        jieba_op.add_word('北京清华大学', freq='test')

    # Test add_dict with space
    jieba_op = JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.MIX)
    with pytest.raises(ValueError, match="user dict file      is not exist"):
        jieba_op.add_dict("    ")

    # Test jieba tokenizer mode is JiebaMode.HMM, data is 1234
    data = 1234
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    with pytest.raises(RuntimeError, match="JiebaTokenizerOp: the input shape should be scalar and the input "
                                           "datatype should be string."):
        _ = jieba_op(data)

    # Test jieba tokenizer mode is JiebaMode.HMM, data is ["hhh", "哈喽", "world"]
    data = ["hhh", "哈喽", "world"]
    jieba_op = nlp.JiebaTokenizer(JIEBATOKENIZER_HMM_FILE, JIEBATOKENIZER_MP_FILE, mode=JiebaMode.HMM)
    with pytest.raises(RuntimeError, match="JiebaTokenizerOp: the input shape should be scalar and the input "
                                           "datatype should be string."):
        _ = jieba_op(data)
