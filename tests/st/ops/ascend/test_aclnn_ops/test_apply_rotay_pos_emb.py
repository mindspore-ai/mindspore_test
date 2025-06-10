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
import numpy as np
import pytest

import mindspore.nn as nn
import mindspore as ms
from mindspore import context, Tensor
from mindspore.common.np_dtype import bfloat16
from mindspore.ops.operations import ApplyRotaryPosEmbExt

from tests.mark_utils import arg_mark

def get_ms_dtype(query_dtype):
    if query_dtype == np.float32:
        ms_dtype = ms.float32
    elif query_dtype == np.float16:
        ms_dtype = ms.float16
    elif query_dtype == bfloat16:
        ms_dtype = ms.bfloat16
    return ms_dtype

class RotaryEmbedding(nn.Cell):
    # cosFormat=0  shape是[maxSeqLen, headDim]，    cos/sin不交替
    # cosFormat=1  shape是[maxSeqLen, headDim]，    cos/sin交替
    # cosFormat=2  shape是[batch*seqLen, headDim]， cos/sin不交替
    # cosFormat=3  shape是[batch*seqLen, headDim]， cos/sin交替
    def __init__(self, dim, batch_size, base=10000, max_seq_len=2048, cos_dtype=np.float32, cos_format=0):
        super(RotaryEmbedding, self).__init__()
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2).astype(np.float32) * (1 / dim)))
        t = np.arange(1, max_seq_len+1, dtype=inv_freq.dtype)
        freqs = np.outer(t, inv_freq)
        if cos_format == 0 or cos_format == 2:
            emb = np.concatenate((freqs, freqs), axis=-1)
        else:
            freqs = np.expand_dims(freqs, 2)
            emb = np.concatenate((freqs, freqs), axis=-1)
            emb = emb.reshape(max_seq_len, dim)
        emb = np.tile(emb[np.newaxis, :, np.newaxis, :], (batch_size, 1, 1, 1))
        self.cos_np = np.cos(emb).astype(cos_dtype)
        self.sin_np = np.sin(emb).astype(cos_dtype)
        self.cos = Tensor(np.cos(emb), dtype=get_ms_dtype(cos_dtype))
        self.sin = Tensor(np.sin(emb), dtype=get_ms_dtype(cos_dtype))
        self.apply_rotary_pos_emb = ApplyRotaryPosEmbExt()
        self.dim = dim
        self.cos_format = cos_format

    def construct(self, query, key):
        query_embed, key_embed = self.apply_rotary_pos_emb(query, key, self.cos, self.sin, 1)
        return query_embed, key_embed

    def rotate_half1(self, x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return np.concatenate((-x2, x1), axis=-1)

    def cal_truth_numpy(self, query, key, position_ids, query_dtype, cos_format):
        query = query.astype(query_dtype).astype(np.float32)
        key = key.astype(query_dtype).astype(np.float32)
        cos1 = self.cos_np.astype(query_dtype).astype(np.float32)
        sin1 = self.sin_np.astype(query_dtype).astype(np.float32)
        cos1 = cos1.transpose((0, 2, 1, 3))
        sin1 = sin1.transpose((0, 2, 1, 3))
        query_embed = (query * cos1) + (self.rotate_half1(query) * sin1)
        key_embed = (key * cos1) + (self.rotate_half1(key) * sin1)
        query_embed = query_embed.astype(np.float32)
        key_embed = key_embed.astype(np.float32)
        return query_embed, key_embed

def run(net, seqLen, batch, num_head_q, num_head_k, hidden_dim, max_seq_len, query_dtype, pos_dtype, ndim=3,
        cos_format=0):
    if ndim == 3:
        query = np.random.rand(batch, seqLen, num_head_q * hidden_dim).astype(np.float32)
        key = np.random.rand(batch, seqLen, num_head_k * hidden_dim).astype(np.float32)
    else:
        query = np.random.rand(batch, seqLen, num_head_q, hidden_dim).astype(np.float32)
        key = np.random.rand(batch, seqLen, num_head_k, hidden_dim).astype(np.float32)

    query2 = query.copy()
    key2 = key.copy()
    query_tmp = Tensor(query, dtype=get_ms_dtype(query_dtype))
    key_tmp = Tensor(key, dtype=get_ms_dtype(query_dtype))
    #[B, N, S, D]
    query_embed1, key_embed1 = net(query_tmp, key_tmp)
    query_embed1 = query_embed1.astype(ms.float32).asnumpy()
    key_embed1 = key_embed1.astype(ms.float32).asnumpy()
    query1 = query2.reshape((batch, seqLen, num_head_q, hidden_dim)).transpose((0, 2, 1, 3))
    key1 = key2.reshape((batch, seqLen, num_head_k, hidden_dim)).transpose((0, 2, 1, 3))
    #[B, N, S, D]->B, S, N, D
    print("query1.shape:", query1.shape)
    query_embed2, key_embed2 = net.cal_truth_numpy(query1, key1, position_ids, query_dtype, cos_format)
    query_embed2 = query_embed2.transpose((0, 2, 1, 3)).reshape(query.shape)
    key_embed2 = key_embed2.transpose((0, 2, 1, 3)).reshape(key.shape)

    if cos_format == 1 or cos_format == 3:
        tmp_shape1, tmp_shape2 = query_embed2.shape, key_embed2.shape
        query_embed2 = query_embed2.reshape(-1, 2, hidden_dim // 2).transpose((0, 2, 1)).reshape(tmp_shape1)
        key_embed2 = key_embed2.reshape(-1, 2, hidden_dim // 2).transpose((0, 2, 1)).reshape(tmp_shape2)
    print("zhanghan query_embed1 query_embed2")
    np.testing.assert_allclose(query_embed1, query_embed2, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(key_embed1, key_embed2, rtol=1e-2, atol=1e-2)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('query_dtype', [np.float16])
@pytest.mark.parametrize('cos_dtype', [np.float16])
@pytest.mark.parametrize('cos_format', [2])
@pytest.mark.parametrize('batch_size', [1, 16])
@pytest.mark.parametrize('seq_len', [1, 256, 512, 1024])
@pytest.mark.parametrize('num_head', [32])
def test_rope_float16(query_dtype, cos_dtype, cos_format, batch_size, seq_len, num_head):
    '''
    Feature:aclnnApplyRotaryPosEmb kernel.
    Description: test for ApplyRotaryPosEmbExt ops.
    Expectation:should pass for all testcases.
    '''
    ndim = 4
    hidden_dim = 128
    base = 10000
    max_seq_len = seq_len
    ms.set_context(device_target="Ascend", mode=context.GRAPH_MODE)
    ms.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    #ms.set_context(save_graphs=True, save_graphs_path="./zhanghanIR")
    net = RotaryEmbedding(hidden_dim, batch_size, base, max_seq_len, cos_dtype, cos_format)
    run(net, seq_len, batch_size, num_head, num_head, hidden_dim, max_seq_len, query_dtype, np.int32, ndim, cos_format)
