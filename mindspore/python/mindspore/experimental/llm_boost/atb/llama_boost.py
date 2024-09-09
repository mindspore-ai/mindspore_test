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
"""llm boost"""
import json
import mindspore.common.dtype as mstype
from mindspore.experimental.llm_boost.atb.boost_base import AtbBoostBase
from mindspore._c_expression import LlmBoostBinder
from mindspore.experimental.llm_boost.register import LlmBoostRegister, LlmBoostType


@LlmBoostRegister.register(LlmBoostType.BUILDIN, "Llama")
class LlamaBoost(AtbBoostBase):
    """LlamaBoost class"""

    def __init__(self, config):
        super().__init__(config)
        self.in_tensor_length = 13
        self.acl_encoder_operation_inputs = [None] * self.in_tensor_length
        self.acl_decoder_operation_inputs = [None] * self.in_tensor_length
        self.atb_encoder_operation = LlmBoostBinder(
            "ATB", "llama_parallel_DecoderModel")
        self.atb_decoder_operation = LlmBoostBinder(
            "ATB", "llama_parallel_DecoderModel")

    def init(self):
        """set param"""
        coder_param = {
            "rmsNormEps": self.config.rms_norm_eps,
            "numAttentionHeadsPerRank": self.config.num_heads // self.device_num,
            "hiddenSizePerAttentionHead": self.head_dim,
            "numHiddenLayers": self.num_layers,
            "numKeyValueHeadsPerRank": self.n_kv_heads // self.device_num,
            "skipWordEmbedding": False,
            "isFA": False,
            "isBF16": self.dtype == mstype.bfloat16,
            "packQuantType": [[1, 1] for _ in range(self.num_layers)],
            "linearQuantType": [[0, -1, -1, 0, 0, -1, 0] for _ in range(self.num_layers)],
            "linearTransposeType": [[1, -1, -1, 1, 1, -1, 1] for i in range(self.num_layers)],
            "isEmbeddingParallel": False,
            "isLmHeadParallel": not self.config.parallel_config.vocab_emb_dp,
            "lmHeadTransposeType": 1,
            "supportSwiGLU": True,
            "kvQuant": self.kv_quant is not None,
            "rank": self.rank_id,
            "worldSize": self.device_num,
            "backend": "lccl",
            "rankTableFile": "",
            "positionEmbeddingType": self.position_embedding_type,
            "hiddenSize": self.config.hidden_size,
            "gemma": False,
            "enableAddNorm": True,
            "supportCompressHead": False,
        }
        encoder_param = {
            **coder_param, "isPrefill": True,
            "supportLcoc": True,
            "supportSpeculate": False,
            "skipWordEmbedding": False
        }
        decoder_param = {
            **coder_param, "isPrefill": False, "supportLcoc": False,
            "supportSpeculate": False
        }
        self.atb_encoder_operation.init(json.dumps({**encoder_param}))
        self.atb_decoder_operation.init(json.dumps({**decoder_param}))

    def _prepare_inputs(
            self,
            prefill=None,
            input_ids=None,
            position_ids=None,
            cos_embed=None,
            sin_embed=None,
            attention_mask=None,
            block_tables=None,
            slots=None,
            input_lengths=None,
            lm_head_indices=None,
            seqLen=None,
            **kwargs
    ):
        """prepare inputs"""
        self.acl_param = json.dumps({
            "seqLen": seqLen,
        })
        self.acl_decoder_operation_inputs[0] = self.cast(
            input_ids, mstype.int64)
        self.acl_decoder_operation_inputs[1] = self.placeholder
        self.acl_decoder_operation_inputs[2] = self.cast(
            position_ids, mstype.int32)
        self.acl_decoder_operation_inputs[3] = cos_embed
        self.acl_decoder_operation_inputs[4] = sin_embed
        self.acl_decoder_operation_inputs[5] = attention_mask
        self.acl_decoder_operation_inputs[6] = block_tables
        self.acl_decoder_operation_inputs[7] = slots
        self.acl_decoder_operation_inputs[8] = self.placeholder
        self.acl_decoder_operation_inputs[9] = self.placeholder
        self.acl_decoder_operation_inputs[10] = self.placeholder
        self.acl_decoder_operation_inputs[11] = self.cast(
            input_lengths, mstype.int32)
        self.acl_decoder_operation_inputs[12] = self.cast(
            lm_head_indices, mstype.int64)
        return self.acl_decoder_operation_inputs, self.acl_param
