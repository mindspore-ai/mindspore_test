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
# ============================================================================

"""
transform huggingface model to mindspore safetensor.
"""
import os
import json
import gc
import numpy as np
from tqdm import tqdm

import mindspore as ms
from mindspore.communication.management import get_rank
from weight_processor import BaseWeightProcessor


def convert_np_to_ms_dtype(value):
    """convert_np_to_ms_dtype"""
    if value.dtype == np.int8:
        value_dtype = ms.int8
    elif value.dtype == np.int32:
        value_dtype = ms.int32
    elif value.dtype == np.int64:
        value_dtype = ms.int64
    elif value.dtype == np.float64:
        value_dtype = ms.float64
    elif value.dtype == np.float32:
        value_dtype = ms.float32
    else:
        value_dtype = ms.bfloat16
    return value_dtype


class DeepseekV3WeightProcessor(BaseWeightProcessor):
    r"""
    Provide DeepseekV3/R1 Model weight load and shards.
    Args:
        config (DeepseekV3/R1Config): The config of DeepseekV3/R1 model.
        network (InferenceDeepseekV3ForCausalLM): The network of DeepseekV3/R1.

    """

    def __init__(self, config, network, is_quant):
        super().__init__(config, network, is_quant)
        self.num_layers = self.config.model.model_config.num_layers
        self.expert_num = self.config.moe_config.expert_num
        self.num_router_experts = self.config.moe_config.expert_num if self.config.moe_config.expert_num else 1

    def infer_trans_rope_weight(self, weight, qk_rope_head_dim):
        """process rope router weight"""
        w1 = weight[..., -qk_rope_head_dim::2, :]
        w2 = weight[..., -qk_rope_head_dim + 1::2, :]
        weight[..., -qk_rope_head_dim:, :] = np.concatenate([w1, w2], axis=-2)
        return weight

    def convert_weight_name(self, weight_name: str):
        """replace weight name"""
        weight_name = weight_name.replace('embed_tokens.weight', 'tok_embeddings.embedding_weight')
        weight_name = weight_name.replace('.self_attn.q_a_proj.', '.attention.q2l_proj.')
        weight_name = weight_name.replace('.self_attn.q_a_layernorm.', '.attention.lq_norm.')
        weight_name = weight_name.replace('.self_attn.q_b_proj.', '.attention.l2q_proj.')
        weight_name = weight_name.replace('.self_attn.kv_a_proj_with_mqa.', '.attention.kv2l.')
        weight_name = weight_name.replace('.self_attn.kv_a_layernorm.', '.attention.lkv_norm.')
        weight_name = weight_name.replace('.self_attn.kv_b_proj.', '.attention.lkv2kv.')
        weight_name = weight_name.replace('.self_attn.o_proj.', '.attention.wo.')
        weight_name = weight_name.replace('mlp.gate_proj.', 'feed_forward.w1.')
        weight_name = weight_name.replace('mlp.down_proj.', 'feed_forward.w2.')
        weight_name = weight_name.replace('mlp.up_proj.', 'feed_forward.w3.')
        weight_name = weight_name.replace('mlp.experts.', 'feed_forward.routed_experts.ffn.')
        weight_name = weight_name.replace('mlp.shared_experts.gate_proj.', 'feed_forward.shared_experts.w1.')
        weight_name = weight_name.replace('mlp.shared_experts.down_proj.', 'feed_forward.shared_experts.w2.')
        weight_name = weight_name.replace('mlp.shared_experts.up_proj.', 'feed_forward.shared_experts.w3.')
        weight_name = weight_name.replace('mlp.gate.weight', 'feed_forward.routed_experts.router.dense.weight')
        weight_name = weight_name.replace('mlp.gate.e_score_correction_bias',
                                          'feed_forward.routed_experts.router.e_score_correction_bias')
        weight_name = weight_name.replace('.input_layernorm.', '.attention_norm.')
        weight_name = weight_name.replace('.post_attention_layernorm.', '.ffn_norm.')
        weight_name = weight_name.replace('model.norm.weight', 'model.norm_out.weight')

        return weight_name


    def infer_process_moe_routed_expert_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """process moe router expert weight"""
        ffn_concat = self.config.model.model_config.ffn_concat

        # router expert dense
        router_dense_hf_name = f"model.layers.{layer_id}.mlp.gate.weight"
        router_dense_ms_name = self.convert_weight_name(router_dense_hf_name)
        router_dense_ms_param, _ = self.get_safetensor_from_file(router_dense_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[router_dense_ms_name] = ms.Parameter(
            ms.from_numpy(router_dense_ms_param).astype(ms.bfloat16),
            name=router_dense_ms_name, requires_grad=False)

        # e_score_correction_bias
        e_score_correction_bias_hf_name = f"model.layers.{layer_id}.mlp.gate.e_score_correction_bias"
        e_score_correction_bias_ms_name = self.convert_weight_name(e_score_correction_bias_hf_name)
        e_score_correction_bias_ms_param, _ = self.get_safetensor_from_file(e_score_correction_bias_hf_name, src_hf_dir,
                                                                            hf_weight_map)
        self.parameter_dict[e_score_correction_bias_ms_name] = ms.Parameter(
            ms.from_numpy(e_score_correction_bias_ms_param).astype(ms.float32),
            name=e_score_correction_bias_ms_name, requires_grad=False)

        w1_list = []
        w2_list = []
        w3_list = []

        w1_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w1.weight"
        w2_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w2.weight"
        w3_ms_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w3.weight"

        for index in range(0, self.num_router_experts):
            w1_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.gate_proj.weight"
            w1_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w1_hf_name, src_hf_dir, hf_weight_map,
                                                                          split_axis=0)

            w2_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.down_proj.weight"
            w2_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w2_hf_name, src_hf_dir, hf_weight_map,
                                                                          split_axis=1)

            w3_hf_name = f"model.layers.{layer_id}.mlp.experts.{index}.up_proj.weight"
            w3_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w3_hf_name, src_hf_dir, hf_weight_map,
                                                                          split_axis=0)

            w1_list.append(w1_ms_param)
            w2_list.append(w2_ms_param)
            w3_list.append(w3_ms_param)

        w1_ms_stack_param = np.stack(w1_list, axis=0)
        w2_ms_stack_param = np.stack(w2_list, axis=0)
        w3_ms_stack_param = np.stack(w3_list, axis=0)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.routed_experts.ffn.w_gate_hidden.weight"
            w_gate_hidden_np = np.concatenate([w1_ms_stack_param, w3_ms_stack_param], axis=1)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_np).permute(0, 2, 1).astype(dtype=ms.bfloat16)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param,
                                                                   name=w_gate_hidden_name,
                                                                   requires_grad=False)
        else:
            w1_ms_stack_param = ms.from_numpy(w1_ms_stack_param).permute(0, 2, 1).astype(ms.bfloat16)
            self.parameter_dict[w1_ms_name] = ms.Parameter(w1_ms_stack_param,
                                                           name=w1_ms_name,
                                                           requires_grad=False)

            w3_ms_stack_param = ms.from_numpy(w3_ms_stack_param).permute(0, 2, 1).astype(ms.bfloat16)
            self.parameter_dict[w3_ms_name] = ms.Parameter(w3_ms_stack_param,
                                                           name=w3_ms_name,
                                                           requires_grad=False)

        w2_ms_stack_param = ms.from_numpy(w2_ms_stack_param).permute(0, 2, 1).astype(ms.bfloat16)
        self.parameter_dict[w2_ms_name] = ms.Parameter(w2_ms_stack_param,
                                                       name=w2_ms_name,
                                                       requires_grad=False)

    def get_moe_shared_expert_weight(self, w1_hf_name, w2_hf_name, w3_hf_name, src_hf_dir, hf_weight_map):
        w1_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w1_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)
        w2_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w2_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=1)
        w3_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w3_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)

        return w1_ms_param, w2_ms_param, w3_ms_param

    def infer_process_moe_shared_expert_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process moe shared expert ffn weight"""
        ffn_concat = self.config.model.model_config.ffn_concat
        w1_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight"
        w2_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.down_proj.weight"
        w3_hf_name = f"model.layers.{layer_id}.mlp.shared_experts.up_proj.weight"

        w1_ms_name = self.convert_weight_name(w1_hf_name)
        w2_ms_name = self.convert_weight_name(w2_hf_name)
        w3_ms_name = self.convert_weight_name(w3_hf_name)

        w1_ms_param, w2_ms_param, w3_ms_param = self.get_moe_shared_expert_weight(w1_hf_name, w2_hf_name, w3_hf_name,
                                                                                  src_hf_dir, hf_weight_map)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.shared_experts.w_gate_hidden.weight"
            w_gate_hidden_np = np.concatenate([w1_ms_param, w3_ms_param], axis=0)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_np).astype(ms.bfloat16)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param,
                                                                   name=w_gate_hidden_name,
                                                                   requires_grad=False)
        else:
            self.parameter_dict[w1_ms_name] = ms.Parameter(ms.from_numpy(w1_ms_param).astype(ms.bfloat16),
                                                           name=w1_ms_name,
                                                           requires_grad=False)
            self.parameter_dict[w3_ms_name] = ms.Parameter(ms.from_numpy(w3_ms_param).astype(ms.bfloat16),
                                                           name=w3_ms_name,
                                                           requires_grad=False)
        self.parameter_dict[w2_ms_name] = ms.Parameter(ms.from_numpy(w2_ms_param).astype(ms.bfloat16),
                                                       name=w2_ms_name,
                                                       requires_grad=False)

    def infer_process_dense_ffn_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process dense ffn weight"""

        ffn_concat = self.config.model.model_config.ffn_concat

        w1_hf_name = f"model.layers.{layer_id}.mlp.gate_proj.weight"
        w1_ms_name = self.convert_weight_name(w1_hf_name)
        w1_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w1_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)

        w2_hf_name = f"model.layers.{layer_id}.mlp.down_proj.weight"
        w2_ms_name = self.convert_weight_name(w2_hf_name)
        w2_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w2_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=1)

        w3_hf_name = f"model.layers.{layer_id}.mlp.up_proj.weight"
        w3_ms_name = self.convert_weight_name(w3_hf_name)
        w3_ms_param, _ = self.get_safetensor_from_file_split_tp_group(w3_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)

        if ffn_concat:
            w_gate_hidden_name = f"model.layers.{layer_id}.feed_forward.w_gate_hidden.weight"
            w_gate_hidden_np = np.concatenate([w1_ms_param, w3_ms_param], axis=0)
            w_gate_hidden_param = ms.from_numpy(w_gate_hidden_np).astype(ms.bfloat16)
            self.parameter_dict[w_gate_hidden_name] = ms.Parameter(w_gate_hidden_param,
                                                                   name=w_gate_hidden_name,
                                                                   requires_grad=False)
        else:
            self.parameter_dict[w1_ms_name] = ms.Parameter(ms.from_numpy(w1_ms_param).astype(ms.bfloat16),
                                                           name=w1_ms_name,
                                                           requires_grad=False)
            self.parameter_dict[w3_ms_name] = ms.Parameter(ms.from_numpy(w3_ms_param).astype(ms.bfloat16),
                                                           name=w3_ms_name,
                                                           requires_grad=False)

        self.parameter_dict[w2_ms_name] = ms.Parameter(ms.from_numpy(w2_ms_param).astype(ms.bfloat16),
                                                       name=w2_ms_name,
                                                       requires_grad=False)

    def infer_process_attention_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process attention weight"""
        num_heads = self.config.model.model_config.num_heads
        kv_lora_rank = self.config.model.model_config.kv_lora_rank
        qk_rope_head_dim = self.config.model.model_config.qk_rope_head_dim
        v_head_dim = self.config.model.model_config.v_head_dim
        qk_nope_head_dim = self.config.model.model_config.qk_nope_head_dim

        rope_dim = qk_rope_head_dim + qk_nope_head_dim
        kv_head_dim = kv_lora_rank + qk_rope_head_dim

        qkv_concat = self.config.model.model_config.qkv_concat
        # q2l_proj
        q2l_proj_hf_name = f"model.layers.{layer_id}.self_attn.q_a_proj.weight"
        q2l_proj_ms_name = self.convert_weight_name(q2l_proj_hf_name)
        q_a_proj_ms_param, _ = self.get_safetensor_from_file(q2l_proj_hf_name, src_hf_dir, hf_weight_map)

        # kv2l
        kv2l_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_proj_with_mqa.weight"
        kv2l_ms_name = self.convert_weight_name(kv2l_hf_name)
        kv2l_ms_param, _ = self.get_safetensor_from_file(kv2l_hf_name, src_hf_dir, hf_weight_map)
        kv2l_ms_param = kv2l_ms_param.reshape(kv_head_dim, -1)
        kv2l_ms_param = self.infer_trans_rope_weight(kv2l_ms_param, qk_rope_head_dim)
        if qkv_concat:
            wqkv2l_weight = np.concatenate((q_a_proj_ms_param, kv2l_ms_param), 0)
            wqkv2l_weight_name = f"model.layers.{layer_id}.attention.qkv2l.weight"
            self.parameter_dict[wqkv2l_weight_name] = ms.Parameter(ms.from_numpy(wqkv2l_weight).astype(ms.bfloat16),
                                                                   name=wqkv2l_weight_name,
                                                                   requires_grad=False)
        else:
            self.parameter_dict[q2l_proj_ms_name] = ms.Parameter(ms.from_numpy(q_a_proj_ms_param).astype(ms.bfloat16),
                                                                 name=q2l_proj_ms_name,
                                                                 requires_grad=False)
            self.parameter_dict[kv2l_ms_name] = ms.Parameter(ms.from_numpy(kv2l_ms_param).astype(ms.bfloat16),
                                                             name=kv2l_ms_name,
                                                             requires_grad=False)
        # lq_norm
        lq_norm_hf_name = f"model.layers.{layer_id}.self_attn.q_a_layernorm.weight"
        lq_norm_ms_name = self.convert_weight_name(lq_norm_hf_name)
        lq_norm_ms_param, _ = self.get_safetensor_from_file(lq_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[lq_norm_ms_name] = ms.Parameter(ms.from_numpy(lq_norm_ms_param).astype(ms.bfloat16),
                                                            name=lq_norm_ms_name,
                                                            requires_grad=False)

        # l2q_proj
        l2q_proj_hf_name = f"model.layers.{layer_id}.self_attn.q_b_proj.weight"
        l2q_proj_ms_name = self.convert_weight_name(l2q_proj_hf_name)
        l2q_proj_ms_param, _ = self.get_safetensor_from_file(l2q_proj_hf_name, src_hf_dir, hf_weight_map)
        l2q_proj_ms_param = l2q_proj_ms_param.reshape(num_heads, rope_dim, -1)
        l2q_proj_ms_param = self.infer_trans_rope_weight(l2q_proj_ms_param, qk_rope_head_dim)
        l2q_proj_ms_param = l2q_proj_ms_param.reshape(num_heads * rope_dim, -1)
        l2q_proj_ms_param = self.split_weight_by_rank(l2q_proj_ms_param, split_axis=0)
        self.parameter_dict[l2q_proj_ms_name] = ms.Parameter(
            ms.from_numpy(l2q_proj_ms_param).astype(ms.bfloat16),
            name=l2q_proj_ms_name,
            requires_grad=False)

        # lkv_norm
        lkv_norm_hf_name = f"model.layers.{layer_id}.self_attn.kv_a_layernorm.weight"
        lkv_norm_ms_name = self.convert_weight_name(lkv_norm_hf_name)
        lkv_norm_ms_param, _ = self.get_safetensor_from_file(lkv_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[lkv_norm_ms_name] = ms.Parameter(
            ms.from_numpy(lkv_norm_ms_param).astype(ms.bfloat16),
            name=lkv_norm_ms_name,
            requires_grad=False)

        # lkv2kv
        lkv2kv_hf_name = f"model.layers.{layer_id}.self_attn.kv_b_proj.weight"
        lkv2kv_ms_name = self.convert_weight_name(lkv2kv_hf_name)
        lkv2kv_ms_param, _ = self.get_safetensor_from_file(lkv2kv_hf_name, src_hf_dir, hf_weight_map)
        lkv2kv_head = qk_nope_head_dim + v_head_dim
        lkv2kv_ms_param = lkv2kv_ms_param.reshape(num_heads, lkv2kv_head, -1)
        value_k_nope, value_v = lkv2kv_ms_param[:, :qk_nope_head_dim, :], lkv2kv_ms_param[:, qk_nope_head_dim:, :]

        # value_k_nope
        value_k_nope = value_k_nope.reshape(-1, value_k_nope.shape[-1])
        value_k_nope = self.split_weight_by_rank(value_k_nope, split_axis=0)
        name_k_nope = lkv2kv_ms_name.replace(".attention.lkv2kv.", ".attention.lkv2kv_k_nope.")
        self.parameter_dict[name_k_nope] = ms.Parameter(ms.from_numpy(value_k_nope).astype(ms.bfloat16),
                                                        name=name_k_nope,
                                                        requires_grad=False)
        # value_v
        value_v = value_v.reshape(-1, value_v.shape[-1])
        value_v = self.split_weight_by_rank(value_v, split_axis=0)
        name_v = lkv2kv_ms_name.replace(".attention.lkv2kv.", ".attention.lkv2kv_v.")
        self.parameter_dict[name_v] = ms.Parameter(ms.from_numpy(value_v).astype(ms.bfloat16),
                                                   name=name_v,
                                                   requires_grad=False)

        # wo
        wo_hf_name = f"model.layers.{layer_id}.self_attn.o_proj.weight"
        wo_ms_name = self.convert_weight_name(wo_hf_name)
        wo_ms_param, _ = self.get_safetensor_from_file(wo_hf_name, src_hf_dir, hf_weight_map)
        wo_ms_param = self.split_weight_by_rank(wo_ms_param, split_axis=1)
        self.parameter_dict[wo_ms_name] = ms.Parameter(ms.from_numpy(wo_ms_param).astype(ms.bfloat16),
                                                       name=wo_ms_name,
                                                       requires_grad=False)

    def infer_process_norm_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer process attention weight"""
        # attention_norm
        attention_norm_hf_name = f"model.layers.{layer_id}.input_layernorm.weight"
        attention_norm_ms_name = self.convert_weight_name(attention_norm_hf_name)
        attention_norm_ms_param, _ = self.get_safetensor_from_file(attention_norm_hf_name,
                                                                   src_hf_dir,
                                                                   hf_weight_map)
        self.parameter_dict[attention_norm_ms_name] = ms.Parameter(
            ms.from_numpy(attention_norm_ms_param).astype(ms.bfloat16),
            name=attention_norm_ms_name,
            requires_grad=False)

        # ffn_norm
        ffn_norm_hf_name = f"model.layers.{layer_id}.post_attention_layernorm.weight"
        ffn_norm_ms_name = self.convert_weight_name(ffn_norm_hf_name)
        ffn_norm_ms_param, _ = self.get_safetensor_from_file(ffn_norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[ffn_norm_ms_name] = ms.Parameter(
            ms.from_numpy(ffn_norm_ms_param).astype(ms.bfloat16),
            name=ffn_norm_ms_name,
            requires_grad=False)

    def infer_convert_outer_weight(self, src_hf_dir, hf_weight_map):
        """convert weight not in model"""
        embed_tokens_hf_name = "model.embed_tokens.weight"
        embed_tokens_ms_name = self.convert_weight_name(embed_tokens_hf_name)
        np_data, _ = self.get_safetensor_from_file(embed_tokens_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[embed_tokens_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.bfloat16),
                                                                 name=embed_tokens_ms_name,
                                                                 requires_grad=False)

        norm_hf_name = "model.norm.weight"
        norm_ms_name = self.convert_weight_name(norm_hf_name)
        np_data, _ = self.get_safetensor_from_file(norm_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[norm_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.bfloat16),
                                                         name=norm_ms_name,
                                                         requires_grad=False)

        lm_head_hf_name = "lm_head.weight"
        lm_head_ms_name = self.convert_weight_name(lm_head_hf_name)
        if not self.config.parallel_config.vocab_emb_dp:
            np_data, _ = self.get_safetensor_from_file_split_tp_group(lm_head_hf_name, src_hf_dir, hf_weight_map,
                                                                      split_axis=0)
        else:
            np_data, _ = self.get_safetensor_from_file(lm_head_hf_name, src_hf_dir, hf_weight_map)
        self.parameter_dict[lm_head_ms_name] = ms.Parameter(ms.from_numpy(np_data).astype(ms.bfloat16),
                                                            name=lm_head_ms_name,
                                                            requires_grad=False)

    def infer_convert_layer_weight(self, src_hf_dir, layer_id, hf_weight_map):
        """infer convert layer weight"""
        if layer_id >= 3:
            self.infer_process_moe_routed_expert_ffn_weight(src_hf_dir, layer_id, hf_weight_map)
            self.infer_process_moe_shared_expert_ffn_weight(src_hf_dir, layer_id, hf_weight_map)
        else:
            self.infer_process_dense_ffn_weight(src_hf_dir, layer_id, hf_weight_map)

        self.infer_process_attention_weight(src_hf_dir, layer_id, hf_weight_map)
        self.infer_process_norm_weight(src_hf_dir, layer_id, hf_weight_map)

    def load_safetensors_shard(self, src_hf_dir, is_mtp_model=False):
        """deepseek load safetensors and shard """
        rank_id = get_rank()
        param_json_path = ""

        for file in os.listdir(src_hf_dir):
            if file.endswith('index.json'):
                # mtp model do not support quantization, needs to load bf16 weight.
                if ('quant' in file and self.is_quant) or \
                        ('quant' not in file and (not self.is_quant or is_mtp_model)):
                    param_json_path = os.path.join(src_hf_dir, file)
                    with open(param_json_path, "r") as fp:
                        hf_weight_map = json.load(fp)['weight_map']
                    break
            elif file.endswith('_name_map.json'):
                param_json_path = os.path.join(src_hf_dir, file)
                with open(param_json_path, "r") as fp:
                    hf_weight_map = json.load(fp)
                    if hf_weight_map.get('weight_map'):
                        hf_weight_map = hf_weight_map['weight_map']
                break

        if not param_json_path:
            raise ValueError(f"Not found param_json_path in {src_hf_dir}")

        enable_tqdm = rank_id == 0
        mtp_layers = self.config.model.model_config.num_nextn_predict_layers
        start_layer = 0 if not is_mtp_model else self.num_layers
        end_layer = self.num_layers if not is_mtp_model else self.num_layers + mtp_layers

        self.infer_convert_outer_weight(src_hf_dir, hf_weight_map)
        for layer_id in tqdm(range(start_layer, end_layer), desc="Weight loading", disable=not enable_tqdm):
            self.infer_convert_layer_weight(src_hf_dir, layer_id, hf_weight_map)

        param_not_load, ckpt_not_load = ms.load_param_into_net(self.network, self.parameter_dict)
        print("param_not_load: %s, ckpt_not_load: %s" % (str(param_not_load), str(ckpt_not_load)))
        del self.parameter_dict
        gc.collect()
