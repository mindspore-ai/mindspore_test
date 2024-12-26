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
"""AscendNative Llama Boost APIs."""

import os
import numpy as np
from mindspore.common import Tensor, dtype
from mindspore.experimental.llm_boost.ascend_native.llm_boost import LLMBoost
from mindspore.experimental.llm_boost.register import LlmBoostRegister, LlmBoostType


def RoundUp(val: int, align: int) -> int:
    if align == 0:
        return 0
    return -(val // -align) * align


def ConvertTensor(nd_mat: np.ndarray, transpose: bool = True, nd2nz: bool = True) -> np.ndarray:
    """ Transforms tensor format from Nd to Nz """
    if transpose:
        nd_mat = np.transpose(nd_mat)
    if not nd2nz:
        return nd_mat
    block_size = (16, 16)
    r = RoundUp(nd_mat.shape[0], block_size[0])
    c = RoundUp(nd_mat.shape[1], block_size[1])
    r_pad = r - nd_mat.shape[0]
    c_pad = c - nd_mat.shape[1]
    nd_mat = np.pad(nd_mat, ((0, r_pad), (0, c_pad)))
    nz_mat = np.transpose(np.reshape(
        nd_mat, (r, c // block_size[1], block_size[1])), (1, 0, 2))
    nz_mat = nz_mat.reshape(r, c)
    return nz_mat


@LlmBoostRegister.register(LlmBoostType.ASCEND_NATIVE, "Llama")
class LlamaBoostAscendNative(LLMBoost):
    r"""
    Implements an Llama model in a single kernel.
    it forwards the python functions to the C++ binded object
    """
    def _get_from_dict(self, dictionary, name):
        """ internal function to get a specific tensor from the dictionary """
        all_relevant_layers = [value for key, value in dictionary.items() if name in key]
        if all_relevant_layers:
            return all_relevant_layers[0].asnumpy()
        return None

    def _get_quant_triplet_from_dict(self, dictionary, name):
        """ internal function to get a weight triple tensor from the dictionary """
        weights = self._get_from_dict(dictionary, name + "._handler.weight")
        scale = self._get_from_dict(dictionary, name + "._weight_quantizer.scale")
        offset = self._get_from_dict(dictionary, name + "._weight_quantizer.zp_neg")
        return weights, scale, offset

    def _prepare_single_layer(self, ckpt, config, id):
        """ prepares the dictionary of weights of a single layer """
        prefix = 'model.layers.' + str(id)
        is_last = (id == config.num_layers-1)
        layer = 'layers.' + str(id) + '.'
        l_dict = {key: value for key, value in ckpt.items() if layer in key}
        if config.n_kv_heads is None:
            config.n_kv_heads = config.num_heads
        start = 0
        end = config.hidden_size
        kv_start = 0
        kv_end = int(config.hidden_size*config.n_kv_heads/config.num_heads)
        ffn_hid = [value for key, value in l_dict.items() if "w3" in key][0].shape[0]
        ffn_start = 0
        ffn_end = ffn_hid
        rank_size = int(os.getenv('RANK_SIZE', '1'))
        #Emir if (config.parallel_mode != 2): # 2 - AUTO_PARALLEL
        hid_size = end
        kv_hid_size = kv_end
        embed_size = config.vocab_size
        rank_id = int(os.getenv('RANK_ID', '0'))
        if (hid_size % rank_size == 0) and (ffn_hid % rank_size == 0) and (embed_size % rank_size == 0):
            start = int(rank_id * hid_size / rank_size)
            end = int((rank_id + 1) * hid_size / rank_size)
            kv_start = int(rank_id * kv_hid_size / rank_size)
            kv_end = int((rank_id + 1) * kv_hid_size / rank_size)
            ffn_start = int(rank_id * ffn_hid / rank_size)
            ffn_end = int((rank_id + 1) * ffn_hid / rank_size)
        else:
            raise RuntimeError("hidden size and ffn hidden size must be divided by rank size without remainder.  \
                                hidden_size: ", hid_size, " ffn_hidden_size: ", ffn_hid, " rank_size: ", rank_size)
        quant = (self._get_from_dict(l_dict, "_weight_quantizer") is not None)
        unite_qkv = (config.num_heads == config.n_kv_heads)
        self.dictionary[prefix + ".attention_norm.weight"] = \
            Tensor(self._get_from_dict(l_dict, "attention_norm"), dtype=dtype.float16)
        self.dictionary[prefix + ".ffn_norm.weight"] = \
            Tensor(self._get_from_dict(l_dict, "ffn_norm"), dtype=dtype.float16)
        if is_last:
            self.dictionary['lm_head.weight'] = Tensor(ConvertTensor(ckpt['lm_head.weight'].asnumpy()[:, start:end]))

        if not quant:
            self._pack_attn_weights(l_dict, prefix, start, end, kv_start, kv_end, unite_qkv)
            self._pack_ffn_weights(l_dict, prefix, ffn_start, ffn_end)
        else:
            self._pack_attn_quant_weights(l_dict, prefix, start, end, kv_start, kv_end, unite_qkv)
            self._pack_ffn_quant_weights(l_dict, prefix, ffn_start, ffn_end)

    def _pack_attn_weights(self, l_dict, prefix, start, end, kv_start, kv_end, unite_qkv):
        """ prepares the dictionary of weights of an attention block """
        wq = self._get_from_dict(l_dict, "wq")[start:end, :]
        wk = self._get_from_dict(l_dict, "wk")[kv_start:kv_end, :]
        wv = self._get_from_dict(l_dict, "wv")[kv_start:kv_end, :]
        self.dictionary[prefix + ".attention.wo.weight"] = \
            Tensor(ConvertTensor(self._get_from_dict(l_dict, "wo")[:, start:end]))
        if unite_qkv:
            self.dictionary[prefix + ".attention.wqkv.weight"] = Tensor(ConvertTensor(np.concatenate((wq, wk, wv))))
        else:
            self.dictionary[prefix + ".attention.wq.weight"] = Tensor(ConvertTensor(wq))
            self.dictionary[prefix + ".attention.wkv.weight"] = Tensor(ConvertTensor(np.concatenate((wk, wv))))

    def _pack_ffn_weights(self, l_dict, prefix, ffn_start, ffn_end):
        """ prepares the dictionary of weights of an ffn block """
        self.dictionary[prefix + ".feed_forward.w2.weight"] = \
            Tensor(ConvertTensor(self._get_from_dict(l_dict, "w2")[:, ffn_start:ffn_end]))
        w1 = self._get_from_dict(l_dict, "w1")[ffn_start:ffn_end, :]
        w3 = self._get_from_dict(l_dict, "w3")[ffn_start:ffn_end, :]
        self.dictionary[prefix + ".feed_forward.w13.weight"] = Tensor(ConvertTensor(np.concatenate((w1, w3))))

    def _pack_attn_quant_weights(self, l_dict, prefix, start, end, kv_start, kv_end, unite_qkv):
        """ prepares the dictionary of weights of a quantized attention block """
        wq, wq_scale, wq_offset = self._get_quant_triplet_from_dict(l_dict, "wq")
        wk, wk_scale, wk_offset = self._get_quant_triplet_from_dict(l_dict, "wk")
        wv, wv_scale, wv_offset = self._get_quant_triplet_from_dict(l_dict, "wv")
        wo, wo_scale, wo_offset = self._get_quant_triplet_from_dict(l_dict, "wo")
        self.dictionary[prefix + ".attention.wo.weight"] = Tensor(ConvertTensor(wo[:, start:end], nd2nz=False))
        self.dictionary[prefix + ".attention.wo.weight.scale"] = Tensor(wo_scale[start:end])
        self.dictionary[prefix + ".attention.wo.weight.offset"] = Tensor(wo_offset[start:end])

        if unite_qkv:
            self.dictionary[prefix + ".attention.wqkv.weight"] = \
             Tensor(ConvertTensor(np.concatenate((wq[start:end, :], wk[kv_start:kv_end, :], wv[kv_start:kv_end, :])),
                                  nd2nz=False))
            self.dictionary[prefix + ".attention.wqkv.weight.scale"] = \
                Tensor(np.concatenate((wq_scale[start:end], wk_scale[kv_start:kv_end], wv_scale[kv_start:kv_end])))
            self.dictionary[prefix + ".attention.wqkv.weight.offset"] = \
                Tensor(np.concatenate((wq_offset[start:end], wk_offset[kv_start:kv_end], wv_offset[kv_start:kv_end])))
        else:
            self.dictionary[prefix + ".attention.wq.weight"] = Tensor(ConvertTensor(wq[start:end, :], nd2nz=False))
            self.dictionary[prefix + ".attention.wq.weight.scale"] = Tensor(wq_scale[start:end])
            self.dictionary[prefix + ".attention.wq.weight.offset"] = Tensor(wq_offset[start:end])
            self.dictionary[prefix + ".attention.wkv.weight"] = \
                Tensor(ConvertTensor(np.concatenate((wk[kv_start:kv_end, :], wv[kv_start:kv_end, :])), nd2nz=False))
            self.dictionary[prefix + ".attention.wkv.weight.scale"] = \
                Tensor(np.concatenate((wk_scale[kv_start:kv_end], wv_scale[kv_start:kv_end])))
            self.dictionary[prefix + ".attention.wkv.weight.offset"] = \
                Tensor(np.concatenate((wk_offset[kv_start:kv_end], wv_offset[kv_start:kv_end])))

    def _pack_ffn_quant_weights(self, l_dict, prefix, ffn_start, ffn_end):
        """ prepares the dictionary of weights of a quantized ffn block """
        w1, w1_scale, w1_offset = self._get_quant_triplet_from_dict(l_dict, "w1")
        w2, w2_scale, w2_offset = self._get_quant_triplet_from_dict(l_dict, "w2")
        w3, w3_scale, w3_offset = self._get_quant_triplet_from_dict(l_dict, "w3")
        self.dictionary[prefix + ".feed_forward.w2.weight"] = Tensor(ConvertTensor(w2[:, ffn_start:ffn_end],
                                                                                   nd2nz=False))
        self.dictionary[prefix + ".feed_forward.w2.weight.scale"] = Tensor(w2_scale[ffn_start:ffn_end])
        self.dictionary[prefix + ".feed_forward.w2.weight.offset"] = Tensor(w2_offset[ffn_start:ffn_end])

        self.dictionary[prefix + ".feed_forward.w13.weight"] = \
                Tensor(ConvertTensor(np.concatenate((w1[ffn_start:ffn_end, :], w3[ffn_start:ffn_end, :])), nd2nz=False))
        self.dictionary[prefix + ".feed_forward.w13.weight.scale"] = \
                Tensor(np.concatenate((w1_scale[ffn_start:ffn_end], w3_scale[ffn_start:ffn_end])))
        self.dictionary[prefix + ".feed_forward.w13.weight.offset"] = \
            Tensor(np.concatenate((w1_offset[ffn_start:ffn_end], w3_offset[ffn_start:ffn_end])))

    def _prepare_cos_sin_arrays(self, config, theta=10000):
        """ prepares the cosine and sine arrays """
        head_dim = config.hidden_size // config.num_heads
        max_position_embedding = \
            config.max_position_embedding if config.max_position_embedding is not None else config.seq_length
        freqs_base = np.arange(0, head_dim, 2)[: (head_dim // 2)].astype(np.float32)
        freqs = 1.0 / (theta ** (freqs_base / head_dim))
        t = np.arange(0, max_position_embedding, 1).astype(np.float32)
        freqs = np.outer(t, freqs)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = Tensor(np.cos(emb), dtype=dtype.float16)
        sin = np.sin(emb)

        sin[:, :int(emb.shape[1]/2)] = -sin[:, :int(emb.shape[1]/2)]
        self.dictionary['model.cos.weight'] = freqs_cos
        freqs_sin = Tensor(sin, dtype=dtype.float16)
        self.dictionary['model.sin.weight'] = freqs_sin

    def set_weights(self, ckpt_dict):
        """ load the checkpoint """
        self.dictionary = {}
        self.dictionary['model.tok_embeddings.embedding_weight'] = \
            Tensor(ckpt_dict['model.tok_embeddings.embedding_weight'].asnumpy())
        self.dictionary['model.norm_out.weight'] = \
            Tensor(ckpt_dict['model.norm_out.weight'].asnumpy(), dtype=dtype.float16)
        self._prepare_cos_sin_arrays(self.config)
        for layer_id in range(self.config.num_layers):
            self._prepare_single_layer(ckpt_dict, self.config, layer_id)

        self.binder.set_weights_map(self.dictionary)
