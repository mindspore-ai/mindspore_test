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
"""boost base class"""
import numpy as np
import mindspore as ms
from mindspore import ops, Tensor
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype
from mindspore._c_expression import _set_format

from mindspore.common.parameter import Parameter
from mindspore.experimental.llm_boost.utils import get_real_rank, get_real_group_size
from mindspore.common.initializer import Zero


class AttentionMask:
    """attention mask"""

    @classmethod
    def static(cls, max_seq_len, dtype=mstype.float16, need_nz=False):
        """cache mask"""
        bias_cache = Tensor(np.tril(np.ones((max_seq_len, max_seq_len), dtype=np.bool_))).reshape(max_seq_len,
                                                                                                  max_seq_len)
        bias_cache = ~bias_cache
        if dtype == mstype.float16:
            mask_value = Tensor(np.finfo(np.float32).min, mstype.float16)
        else:
            mask_value = Tensor(1)
        attn_mask = ops.masked_fill(Tensor(np.zeros(
            (max_seq_len, max_seq_len)), dtype=mstype.float16), bias_cache, mask_value)
        if need_nz:
            # ND -> NZ
            attn_mask = ops.reshape(attn_mask, (1, max_seq_len, max_seq_len))
            attn_mask = ops.reshape(
                attn_mask, (1, max_seq_len, max_seq_len // 16, 16))
            attn_mask = ops.transpose(attn_mask, (0, 2, 1, 3)).contiguous()
            attn_mask = _set_format(attn_mask, "FRACTAL_NZ")
        return attn_mask


class AtbBoostBase():
    """atb boost base class"""

    def __init__(self, config):
        super().__init__()
        self.is_first_iteration = False
        self.config = config
        self.dtype = config.compute_dtype
        self.num_heads = config.num_heads
        self.num_kv_heads = config.n_kv_heads if config.n_kv_heads else self.num_heads
        self.num_layers = config.num_layers
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads else config.num_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.need_nz = False
        if hasattr(config, "need_nz"):
            self.need_nz = config.need_nz
        self.placeholder = Tensor(np.zeros(1), dtype=self.dtype)
        self.lm_head_indices_fake = Tensor([0], dtype=mstype.int64)
        self.position_embedding_type = "ROPE"
        self.add_norm_enable = True
        self.max_decode_length = self.config.max_decode_length
        self.max_base_len = 128
        self.attn_mask = AttentionMask.static(
            self.max_base_len, dtype=self.dtype, need_nz=self.need_nz)

        self.cast = P.Cast()
        self.reshape = P.Reshape()
        self.kv_quant = None
        self.rank_id = get_real_rank()
        self.device_num = get_real_group_size()

    def _convert_tensor_format_and_dtype(self, tensor, dtype=mstype.float16):
        tensor = self.cast(tensor, dtype=dtype)
        if self.need_nz:
            tensor = _set_format(tensor, "FRACTAL_NZ")
        return tensor

    def set_weights(self, parm_dict, dtype=mstype.float16):
        """set weights for llm boost"""
        embedding_weight_name = "model.tok_embeddings.embedding_weight"
        attention_norm_name = "attention_norm"
        qkv_name = "attention.w_qkv"
        o_name = "attention.wo"
        mlp_norm_name = "ffn_norm"
        mlp_gate_name = "feed_forward.w_gate_hidden"
        mlp_down_name = "feed_forward.w2"
        norm_out_name = "model.norm_out"
        lm_head_name = "lm_head"
        placeholder = Parameter(Tensor(np.zeros(1), dtype=dtype))

        ascend_weight = []
        ascend_weight.append(
            self.cast(parm_dict[embedding_weight_name], dtype))
        for i in range(self.num_layers):
            ascend_weight.append(self._convert_tensor_format_and_dtype(
                parm_dict[f"model.layers.{i}.{attention_norm_name}.weight"], dtype))
            ascend_weight.extend([placeholder] * 3)

            ascend_weight.append(
                self._convert_tensor_format_and_dtype(parm_dict[f"model.layers.{i}.{qkv_name}.weight"], dtype))
            ascend_weight.append(self._convert_tensor_format_and_dtype(parm_dict.get(
                f"model.layers.{i}.{qkv_name}.bias", placeholder), dtype))
            ascend_weight.extend([placeholder] * 16)

            ascend_weight.append(
                self._convert_tensor_format_and_dtype(parm_dict[f"model.layers.{i}.{o_name}.weight"], dtype))
            ascend_weight.append(self._convert_tensor_format_and_dtype(parm_dict.get(
                f"model.layers.{i}.{o_name}.bias", placeholder), dtype))
            ascend_weight.extend([placeholder] * 4)

            ascend_weight.append(
                self._convert_tensor_format_and_dtype(parm_dict[f"model.layers.{i}.{mlp_norm_name}.weight"], dtype))
            ascend_weight.extend([placeholder] * 3)

            ascend_weight.append(
                self._convert_tensor_format_and_dtype(parm_dict[f"model.layers.{i}.{mlp_gate_name}.weight"], dtype))
            ascend_weight.append(self._convert_tensor_format_and_dtype(parm_dict.get(
                f"model.layers.{i}.{mlp_gate_name}.bias", placeholder), dtype))
            ascend_weight.extend([placeholder] * 10)

            ascend_weight.append(
                self._convert_tensor_format_and_dtype(parm_dict[f"model.layers.{i}.{mlp_down_name}.weight"], dtype))
            ascend_weight.append(self._convert_tensor_format_and_dtype(parm_dict.get(
                f"model.layers.{i}.{mlp_down_name}.bias", placeholder), dtype))
            ascend_weight.extend([placeholder] * 4)

        ascend_weight.append(
            self._convert_tensor_format_and_dtype(parm_dict[f"{norm_out_name}.weight"], dtype))
        ascend_weight.append(
            self._convert_tensor_format_and_dtype(parm_dict[f"{lm_head_name}.weight"], dtype))
        self.atb_encoder_operation.set_weights(ascend_weight)
        self.atb_decoder_operation.set_weights(ascend_weight)

    def set_kvcache(self, k_caches=None, v_caches=None):
        """set kv_cache for llm boost"""
        if not k_caches or v_caches:
            if self.need_nz:
                kv_shape = (self.config.num_blocks, self.num_kv_heads*self.head_dim //
                            self.device_num // 16, self.config.block_size, 16)
                k_caches = [_set_format(Parameter(Tensor(
                    shape=kv_shape, dtype=self.dtype, init=Zero())), "FRACTAL_NZ") for _ in range(self.num_layers)]
                v_caches = [_set_format(Parameter(Tensor(
                    shape=kv_shape, dtype=self.dtype, init=Zero())), "FRACTAL_NZ") for _ in range(self.num_layers)]
            else:
                kv_shape = (self.config.num_blocks, self.config.block_size,
                            self.num_kv_heads // self.device_num, self.head_dim)
                k_caches = [Parameter(Tensor(
                    shape=kv_shape, dtype=self.dtype, init=Zero())) for _ in range(self.num_layers)]
                v_caches = [Parameter(Tensor(
                    shape=kv_shape, dtype=self.dtype, init=Zero())) for _ in range(self.num_layers)]

        self.atb_encoder_operation.set_kvcache(k_caches, v_caches)
        self.atb_decoder_operation.set_kvcache(k_caches, v_caches)

    def add_flags(self, is_first_iteration):
        """add_flags."""
        self.is_first_iteration = is_first_iteration

    def _execute_operator(self, acl_inputs, acl_param):
        """execute operator."""
        if self.is_first_iteration:
            acl_model_out = self.atb_encoder_operation.forward(
                acl_inputs, acl_param)
        else:
            acl_model_out = self.atb_decoder_operation.forward(
                acl_inputs, acl_param)
        acl_hidden_state = acl_model_out[0]
        return acl_hidden_state

    def forward(self, boost_inputs):
        r"""
        LlmBoost forward.
        """
        input_ids = boost_inputs["input_ids"]
        position_ids = boost_inputs["position_ids"]
        cos_embed = boost_inputs["cos_embed"]
        sin_embed = boost_inputs["sin_embed"]
        block_tables = boost_inputs["block_tables"]
        slot_mapping = boost_inputs["slot_mapping"]
        batch_valid_length = boost_inputs["batch_valid_length"]
        lm_head_indices = boost_inputs["lm_head_indices"]
        seqLen = boost_inputs["seq_lens"]
        if self.is_first_iteration:
            attention_mask = self.attn_mask
        else:
            position_ids = batch_valid_length - 1
            attention_mask = self.placeholder
            lm_head_indices = self.lm_head_indices_fake

        acl_inputs, acl_param = self._prepare_inputs(prefill=self.is_first_iteration, input_ids=input_ids,
                                                     position_ids=position_ids, cos_embed=cos_embed,
                                                     sin_embed=sin_embed, attention_mask=attention_mask,
                                                     block_tables=block_tables, slots=slot_mapping,
                                                     input_lengths=batch_valid_length, lm_head_indices=lm_head_indices,
                                                     seqLen=seqLen)
        ms.hal.synchronize()
        logits = self._execute_operator(acl_inputs, acl_param)
        logits = self.cast(logits, mstype.float32)
        return logits
