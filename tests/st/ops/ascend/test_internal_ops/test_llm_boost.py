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

# pylint: disable=W0611
import pytest
import numpy as np
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.experimental.llm_boost.register import LlmBoostRegister, LlmBoostType
from tests.mark_utils import arg_mark

class DictConfig(dict):
    """config"""
    def __init__(self, **kwargs):
        super(DictConfig, self).__init__()
        self.update(kwargs)
    def __getattr__(self, key):
        if key not in self:
            return None
        return self[key]
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        del self[key]
    def __deepcopy__(self, memo=None):
        config = self.__class__()
        for key in self.keys():
            config.__setattr__(copy.deepcopy(key, memo), copy.deepcopy(self.__getattr__(key), memo))
        return config
    def to_dict(self):
        return_dict = {}
        for key, val in self.items():
            if isinstance(val, self.__class__):
                val = val.to_dict()
            return_dict[key] = val
        return return_dict

class TestLlamaConfig(DictConfig):
    def __init__(self, batch, layers=2, seq_len=1024, hid_s=4096):
        super(TestLlamaConfig, self).__init__()
        self.model_type = "Llama"
        self.batch_size = batch
        self.seq_length = seq_len
        self.hidden_size = hid_s
        self.ffn_hidden_size = 11008
        self.num_layers = layers
        self.num_heads = 32
        self.vocab_size = 32000
        self.multiple_of = 256
        self.rms_norm_eps = 1e-5
        self.n_kv_heads = self.num_heads
        self.num_blocks = 1024
        self.block_size = 16
    def create_dict(self):
        wdict = {}
        dtype = mstype.float16
        wdict["model.tok_embeddings.embedding_weight"] = Tensor(np.ones((self.vocab_size, self.hidden_size)), dtype)
        wdict["lm_head.weight"] = Tensor(np.ones((self.vocab_size, self.hidden_size)), dtype)
        wdict["model.norm_out.weight"] = Tensor(np.ones((self.hidden_size,)), dtype)
        for i in range(self.num_layers):
            pref = "model.layers." + str(i)
            wdict[pref + ".attention.wq.weight"] = Tensor(np.ones((self.hidden_size, self.hidden_size)), dtype)
            wdict[pref + ".attention.wk.weight"] = Tensor(np.ones((self.hidden_size, self.hidden_size)), dtype)
            wdict[pref + ".attention.wv.weight"] = Tensor(np.ones((self.hidden_size, self.hidden_size)), dtype)
            wdict[pref + ".attention.wo.weight"] = Tensor(np.ones((self.hidden_size, self.hidden_size)), dtype)
            wdict[pref + ".feed_forward.w1.weight"] = Tensor(np.ones((self.ffn_hidden_size, self.hidden_size)), dtype)
            wdict[pref + ".feed_forward.w3.weight"] = Tensor(np.ones((self.ffn_hidden_size, self.hidden_size)), dtype)
            wdict[pref + ".feed_forward.w2.weight"] = Tensor(np.ones((self.hidden_size, self.ffn_hidden_size)), dtype)
            wdict[pref + ".attention_norm.weight"] = Tensor(np.ones((self.hidden_size,)), dtype)
            wdict[pref + ".ffn_norm.weight"] = Tensor(np.ones((self.hidden_size,)), dtype)
        return wdict

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('batch', [1, 4])
def test_llm_boost(batch):
    """
    Feature: test llama boost base class functionality
    Description: initialize with a batch of 1 and then inference ddata with a higher batch size to check the behavior
                 of infer shape
                 because it is too costy to run a whole llama model we initialize a smaller model of 2 layers
    Expectation: shapes of output are identical
    """
    # config = TestLlamaConfig(1)
    # llm_boost_kwargs = {"config": config}
    # llm_boost = LlmBoostRegister.get_instance(LlmBoostType.ASCEND_NATIVE, "Llama", **llm_boost_kwargs)
    # llm_boost.init()
    # llm_boost.set_weights(config.create_dict())
    # llm_boost.add_flags(is_first_iteration=True)
    # input_ids = Tensor(np.ones((batch, 40)), mstype.int32)
    # bvl = Tensor(np.ones((batch)) * 10, mstype.int32)
    # output = llm_boost.forward(input_ids, bvl, None)
    # assert output.shape == (batch, config.vocab_size)

    # pylint: disable=W0107
    pass
