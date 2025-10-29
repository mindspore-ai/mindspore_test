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
"""text transform - sentencepiecetokenizer"""

import os
import pytest
from mindspore.dataset import text
from mindspore.dataset.text import SentencePieceModel, SPieceTokenizerOutType


TEST_DATA_DATASET_FUNC ="../data/dataset/"


VOCAB_FILE = os.path.join(TEST_DATA_DATASET_FUNC, "text_data/testTextFile/textfile/test_sentencepiece/botchan.txt")
DATA_FILE = os.path.join(TEST_DATA_DATASET_FUNC, ("text_data/testTextFile/textfile/testTokenizerData"
                                                  "/SentencePieceTokenizer/sentencepiece_tokenizer.txt"))


def test_sentencepiecetokenizer_operation_01():
    """
    Feature: SentencePieceTokenizer op
    Description: Test SentencePieceTokenizer op with different input types
    Expectation: Successfully tokenize strings using SentencePiece model
    """
    # mode is SentencePieceVocab, out_type=SPieceTokenizerOutType.STRING, input is str
    out_type = SPieceTokenizerOutType.STRING
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 4000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=out_type)
    data = "我爱我的祖国"
    res = tokenizer(data)
    assert (res == ['▁', '我爱我的祖国']).all()

    # mode is SentencePieceVocab, out_type=SPieceTokenizerOutType.STRING, input is list(str)
    out_type = SPieceTokenizerOutType.STRING
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 4000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=out_type)
    data = ["哈喽", "我爱我的祖国"]
    res = []
    for i in data:
        res.append(tokenizer(i))
    assert (res[0] == ['▁', '哈喽']).all()
    assert (res[1] == ['▁', '我爱我的祖国']).all()


def test_sentencepiecetokenizer_exception_01():
    """
    Feature: SentencePieceTokenizer op
    Description: Test SentencePieceTokenizer op with missing or invalid parameters
    Expectation: Raise expected exceptions for missing required parameters
    """
    # Test no mode
    out_type = SPieceTokenizerOutType.STRING
    with pytest.raises(TypeError, match="missing a required argument: 'mode'"):
        text.SentencePieceTokenizer(out_type=out_type)

    # Test no out_type
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 4000, 0.9995, SentencePieceModel.UNIGRAM, {})
    with pytest.raises(TypeError, match="missing a required argument: 'out_type'"):
        text.SentencePieceTokenizer(vocab)

    # Test no para

    with pytest.raises(TypeError, match="missing a required argument: 'mode'"):
        text.SentencePieceTokenizer()

    # mode is SentencePieceVocab, out_type=SPieceTokenizerOutType.STRING, input is int
    out_type = SPieceTokenizerOutType.STRING
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 4000, 0.9995, SentencePieceModel.UNIGRAM, {})
    tokenizer = text.SentencePieceTokenizer(vocab, out_type=out_type)
    data = 1234567
    with pytest.raises(RuntimeError, match="SentencePieceTokenizer: the input shape should be scalar and the input "
                                           "datatype should be string."):
        _ = tokenizer(data)
