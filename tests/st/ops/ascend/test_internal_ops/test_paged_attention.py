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
# ============================================================================

import os
import numpy as np
import math
import pytest

import mindspore as ms
from mindspore.ops.operations.nn_ops import PagedAttention
from mindspore.nn import Cell
from mindspore import context, Profiler
from test_paged_attention_mask import PagedAttentionBase


class PagedAttentionNet(Cell):
    """
    PagedAttention must use correct input name so that framework can save the input tensor on host
    """

    def __init__(self, num_head=0, scale_value=0, kv_head=0):
        super().__init__()
        self.net = PagedAttention(num_head, scale_value, kv_head)

    def construct(self, query, key_cache, value_cache, block_tables, batch_valid_length, antiquant_scale=None,
                  antiquant_offset=None, attn_mask=None, q_seq_lens=None):
        return self.net(query, key_cache, value_cache, block_tables, batch_valid_length, antiquant_scale,
                        antiquant_offset, attn_mask, q_seq_lens)


class PagedAttentionTest(PagedAttentionBase):
    def __init__(self, i_test: dict, mode="gen"):
        super().__init__(i_test, mode)
        i_test["use_asdop"] = i_test.get("use_asdop", False)
        if i_test["is_alibi"]:
            raise Exception("[Error] alibi_mask can only be in PagedAttentionMaskTest")
        if i_test["layout"] == "TH":
            raise Exception("[Error] layout 'TH' can only be in Asdop kernel")
        self.calc_expect_func()
        self.calc_actual_func()
        self.compare()

    def calc_actual_func(self):
        if self.i_test["type"] == "float32":
            q_type = ms.bfloat16
        else:  # float16
            q_type = ms.float16
        if self.i_test["is_quant"]:
            kv_type = ms.int8
        else:  # not quant
            kv_type = q_type

        self.i_construct = {
            "query": None,
            "key_cache": None,
            "value_cache": None,
            "block_tables": None,
            "context_lens": None,
            "antiquant_scale": None,
            "antiquant_offset": None,
            "attn_mask": None,
            "q_seq_lens": None
        }
        # query
        if self.i_test["layout"] == "BSH":
            B, Sq, N, G, D = self.gen["query"].shape
            query = np.reshape(self.gen["query"], newshape=[B, Sq, N * G * D])
        elif self.i_test["layout"] == "TH":
            B, Sq, N, G, D = self.gen["query"].shape
            query = np.reshape(self.gen["query"], newshape=[B * Sq, N * G * D])
        else:  # BNSD
            B, N, G, Sq, D = self.gen["query"].shape
            query = np.reshape(self.gen["query"], newshape=[B, N * G, Sq, D])
        self.i_construct["query"] = ms.Tensor(query, dtype=q_type)
        # key_cache value_cache
        B, MaxSkv, N, _, D = self.gen["key_cache"].shape
        block_size = self.i_test["blk_sz"]
        block_num = B * MaxSkv // block_size
        key_cache = np.reshape(
            self.gen["key_cache"],
            newshape=[block_num, block_size, N, D])
        value_cache = np.reshape(
            self.gen["value_cache"],
            newshape=[block_num, block_size, N, D])
        self.i_construct["key_cache"] = ms.Tensor(key_cache, dtype=kv_type)
        self.i_construct["value_cache"] = ms.Tensor(value_cache, dtype=kv_type)
        # block_tables
        self.i_construct["block_tables"] = ms.Tensor(self.gen["block_tables"])
        # context_lens
        self.i_construct["context_lens"] = ms.Tensor(self.gen["context_lens"])
        # quant
        if "antiquant_scale" in self.gen and "antiquant_offset" in self.gen:
            self.i_construct["antiquant_scale"] = ms.Tensor(self.gen["antiquant_scale"])
            self.i_construct["antiquant_offset"] = ms.Tensor(self.gen["antiquant_offset"])
        # alibi_mask
        if "alibi_mask" in self.gen:
            B, N, G, Sq, MaxSkv = self.gen["alibi_mask"].shape
            alibi_mask = np.reshape(
                self.gen["alibi_mask"],
                newshape=[B, N * G, Sq, MaxSkv])
            self.i_construct["alibi_mask"] = ms.Tensor(alibi_mask, dtype=q_type)
        # attn_mask
        if "attn_mask" in self.gen:
            B, _, _, Sq, MaxSkv = self.gen["attn_mask"].shape
            attn_mask = np.reshape(
                self.gen["attn_mask"],
                newshape=[B, Sq, MaxSkv])
            self.i_construct["attn_mask"] = ms.Tensor(attn_mask, dtype=ms.float16)
            self.i_construct["q_seq_lens"] = ms.Tensor(self.gen["sq"])

        self.i_init = {
            "head_num": self.i_test["nq"],  # dtype: int
            "scale_value": 1 / math.sqrt(self.i_test["d"]),  # dtype: float
            "kv_head_num": self.i_test["nkv"]  # dtype: int
        }

        self.o_ascend = {
            "attention_out": None,  # dtype: tensor
        }

        if "ASCEND_HOME_PATH" not in os.environ:
            os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
        context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
        net = PagedAttentionNet(*tuple(self.i_init.values()))

        if "msprof" in self.i_test and self.i_test["msprof"]:
            profiler = Profiler(start_profile=False, output_path="./profiler", data_simplification=False)
            profiler.start()
            for _ in range(self.i_test["msprof"]):
                self.o_ascend["attention_out"] = net(*tuple(self.i_construct.values()))
            profiler.stop()
            profiler.analyse()
        else:
            self.o_ascend["attention_out"] = net(*tuple(self.i_construct.values()))


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_bnsd():
    """
    Feature: test PagedAttention op in kbk enabling infer_boost
    Description: test PagedAttention op in BNSD.
    Expectation: the result is correct
    """
    i_test = {
        "type": "float32",
        "layout": "BNSD",
        "b": 2,
        "sq": 1,
        "skv": "max",
        "max_skv": 512,
        "nq": 9,
        "nkv": 3,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "is_quant": False,
        "is_alibi": False,
        "is_amask": False,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_fd_long():
    """
    Feature: test PagedAttention op in kbk enabling infer_boost
    Description: test PagedAttention op with long sequence.
    Expectation: the result is correct
    """
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 6,
        "skv": "max",
        "max_skv": 20032,
        "nq": 2,
        "nkv": 1,
        "d": 128,
        "blk_sz": 64,
        "drop_prop": 0.0,
        "is_quant": False,
        "is_alibi": False,
        "is_amask": False,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_quant0():
    """
    Feature: test PagedAttention op in kbk enabling infer_boost
    Description: test PagedAttention op with quant.
    Expectation: the result is correct
    """
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 2,
        "sq": 1,
        "skv": "rand-1",
        "max_skv": 1024,
        "nq": 9,
        "nkv": 3,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "is_quant": True,
        "is_alibi": False,
        "is_amask": False,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_quant1():
    """
    Feature: test PagedAttention op in kbk enabling infer_boost
    Description: test PagedAttention op with quant.
    Expectation: the result is correct
    """
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 1,
        "skv": 63,
        "max_skv": 8192,
        "nq": 32,
        "nkv": 32,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "is_quant": True,
        "is_alibi": False,
        "is_amask": False,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_lookahead0():
    """
    Feature: test PagedAttention op in kbk enabling infer_boost
    Description: test PagedAttention op with lookahead.
    Expectation: the result is correct
    """
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 1,
        "sq": 32,
        "skv": "max",
        "max_skv": 8192,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 128,
        "drop_prop": 0.0,
        "is_quant": False,
        "is_alibi": False,
        "is_amask": True,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_lookahead1():
    """
    Feature: test PagedAttention op in kbk enabling infer_boost
    Description: test PagedAttention op with lookahead.
    Expectation: the result is correct
    """
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 2,
        "sq": 11,
        "skv": "rand-1",
        "max_skv": 1024,
        "nq": 9,
        "nkv": 3,
        "d": 256,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "is_quant": False,
        "is_alibi": False,
        "is_amask": True,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_fake_lookahead():
    """
    Feature: test PagedAttention op in kbk enabling infer_boost
    Description: test PagedAttention op with fake_lookahead.
    Expectation: the result is correct
    """
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 8,
        "sq": 1,
        "skv": "rand-1",
        "max_skv": 1024,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "is_quant": False,
        "is_alibi": False,
        "is_amask": True,
        "blk_sparse": 0,
        "msprof": 0
    }
    PagedAttentionTest(i_test)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
def test_paged_attention_asd_quant():
    """
    Feature: test PagedAttention op in kbk enabling infer_boost
    Description: test PagedAttention op with asdop quant mode.
    Expectation: the result is correct
    """
    i_test = {
        "type": "float16",
        "layout": "BSH",
        "b": 8,
        "sq": 1,
        "skv": 1024,
        "max_skv": 4096,
        "nq": 8,
        "nkv": 8,
        "d": 128,
        "blk_sz": 16,
        "drop_prop": 0.0,
        "is_quant": True,
        "is_alibi": False,
        "is_amask": False,
        "blk_sparse": 0,
        "use_asdop": True,
        "msprof": 0
    }
    PagedAttentionTest(i_test)
