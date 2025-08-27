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

import math
import numpy as np
from dataclasses import dataclass

import mindspore as ms
import mindspore.ops.functional as F

from mindspore import nn, Tensor, mint
from mindspore.common import dtype as mstype
from mindspore.communication.management import init, get_rank, get_group_size, create_group
from mindspore.mint.nn import Linear, SiLU, LayerNorm, Embedding, CrossEntropyLoss
from mindspore.parallel.mpmd.pipeline_parallel import Schedule1F1B, PipelineStage, P2PInfo



@dataclass
class ModelArgs:
    vocab_size: int = 256
    hidden_size: int = 64
    intermediate_size: int = 64
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    seq_length: int = 32
    stage_idx: int = 0
    num_stages: int = 2
    compute_dtype: mstype = mstype.float16


class MockMLP(nn.Cell):
    def __init__(self, hidden_size, intermediate_size, compute_dtype=mstype.float16):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.w1 = Linear(hidden_size, intermediate_size, dtype=compute_dtype)
        self.w2 = Linear(intermediate_size, hidden_size, dtype=compute_dtype)
        self.w3 = Linear(hidden_size, intermediate_size, dtype=compute_dtype)
        self.silu = SiLU(inplace=False)

    def construct(self, x):
        gate = self.silu(self.w1(x))
        hidden = self.w3(x)
        hidden = mint.mul(gate, hidden)
        out = self.w2(hidden)
        return out


class TransformerBlock(nn.Cell):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads):
        super().__init__()
        self.ln1 = LayerNorm((hidden_size,))
        self.attn = MockAttention(num_attention_heads, intermediate_size)
        self.ln2 = LayerNorm((hidden_size,))
        self.mlp = MockMLP(hidden_size, intermediate_size)

    def construct(self, x, mask=None):
        input_x = self.ln1(x)
        h = self.attn(input_x, mask)
        h = input_x + h
        ffn_norm = self.ln2(h)
        ffn_out = self.mlp(ffn_norm)
        h = h + ffn_out
        return h


class MockModel(nn.Cell):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.vocab_size = config.vocab_size
        self.compute_dtype = config.compute_dtype
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.num_attention_heads = config.num_attention_heads
        self.tok_embedding = Embedding(self.vocab_size, self.hidden_size) if config.stage_idx == 0 else None

        self.layers = nn.CellDict()
        division = config.num_hidden_layers // config.num_stages
        residual = config.num_hidden_layers % config.num_stages
        layer_per_stage = [
            division + 1 if stage < residual else division for stage in range(config.num_stages)
        ]
        assert sum(layer_per_stage) == config.num_hidden_layers
        layer_id_start = sum(layer_per_stage[:config.stage_idx])
        layer_id_end = layer_id_start + layer_per_stage[config.stage_idx]
        for layer_id in range(layer_id_start, layer_id_end):
            self.layers[str(layer_id)] = TransformerBlock(self.hidden_size, self.intermediate_size,
                                                          self.num_attention_heads)
        self.ln_f = nn.LayerNorm((self.hidden_size,))

    def construct(self, input_ids, mask=None):
        x = self.tok_embedding(input_ids) if self.tok_embedding is not None else input_ids
        x = F.cast(x, self.compute_dtype)
        for layer in self.layers.values():
            x = layer(x, mask)
        norm_output = self.ln_f(x)
        norm_output = F.cast(norm_output, self.compute_dtype)
        return norm_output


class MockLM(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MockModel(config)
        self.lm_head = Linear(config.hidden_size, config.vocab_size,
                              dtype=config.compute_dtype) if config.stage_idx == config.num_stages - 1 else None

    def construct(self, input_ids, mask=None):
        hidden = self.model(input_ids, mask)
        logits = self.lm_head(hidden) if self.lm_head is not None else hidden
        return logits


class MockAttention(nn.Cell):
    def __init__(self, num_attention_heads, hidden_size, compute_dtype=mstype.float16):
        super().__init__()
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scale = 1 / math.sqrt(self.head_dim)
        self.compute_dtype = compute_dtype

        self.q_proj = Linear(hidden_size, hidden_size, dtype=compute_dtype)
        self.k_proj = Linear(hidden_size, hidden_size, dtype=compute_dtype)
        self.v_proj = Linear(hidden_size, hidden_size, dtype=compute_dtype)
        self.out_proj = Linear(hidden_size, hidden_size, dtype=compute_dtype)

    def transpose_for_scores(self, x):
        bs, seq_len, _ = x.shape
        x = mint.reshape(x, (bs, seq_len, self.num_heads, self.head_dim))
        return x.transpose(0, 2, 1, 3)

    def construct(self, x, mask=None):
        bs, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attn_scores = mint.matmul(q, k.transpose(0, 1, 3, 2)) * self.scale
        if mask is not None:
            attn_scores += (mask * -1e9)
        attn_probs = mint.softmax(attn_scores, dim=-1)

        attn_output = mint.matmul(attn_probs, v)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(bs, seq_len, -1)
        return self.out_proj(attn_output)


def build_model_config():
    model_config = ModelArgs()
    model_config.num_stages = 4
    model_config.num_attention_heads = 8
    model_config.compute_dtype = ms.float16
    return model_config


def build_send_recv_info(stage_index, model_config, per_batch_size, seq_length):
    recv_info_list = []
    send_info_list = []
    if stage_index == 0:
        send_info = P2PInfo(shape=[per_batch_size, seq_length, model_config.hidden_size], dtype=ms.float16)
        send_info_list.append(send_info)
    elif stage_index == model_config.num_stages - 1:
        # for forward recv
        recv_info = P2PInfo(shape=[per_batch_size, seq_length, model_config.hidden_size], dtype=ms.float16)
        recv_info_list.append(recv_info)
    else:
        # for forward recv
        recv_info = P2PInfo(shape=[per_batch_size, seq_length, model_config.hidden_size], dtype=ms.float16)
        recv_info_list.append(recv_info)
        # for forward send
        send_info = P2PInfo(shape=[per_batch_size, seq_length, model_config.hidden_size], dtype=ms.float16)
        send_info_list.append(send_info)
    return send_info_list, recv_info_list

def build_pipeline_stage(model_config, pp_group, send_info_list, recv_info_list):
    stage_kwargs = {
        'stage_num': model_config.num_stages,
        'send_info': send_info_list,
        'recv_info': recv_info_list,
        'group': pp_group
    }
    model = MockLM(model_config)

    # build pipeline_stage
    pipeline_stage = PipelineStage(model, model_config.stage_idx, **stage_kwargs)
    return pipeline_stage

# loss_fn
def calculate_loss(model_outputs, model_targets, micro_batch_num, vocab_size=256):
    def loss_fn(outputs, targets, vocab_size):
        loss_func = CrossEntropyLoss()
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        outputs = mint.reshape(outputs, (-1, vocab_size))
        targets = mint.reshape(targets, (-1,))
        return loss_func(outputs, targets)

    loss = None
    per_micro_size = model_targets.shape[0] // micro_batch_num
    for idx, output in enumerate(model_outputs):
        if idx == 0:
            loss = loss_fn(output, model_targets[idx * per_micro_size: (idx + 1) * per_micro_size],
                           vocab_size=vocab_size)
        else:
            loss += loss_fn(output, model_targets[idx * per_micro_size: (idx + 1) * per_micro_size],
                            vocab_size=vocab_size)
    loss /= micro_batch_num
    return loss


def test_schedule_1f1b_less_micro():
    """
    Feature: Pipeline schedule 1f1b
    Description: Execute 1f1b schedule when micro batch num can not fill up warmup phase
    Expectation: Run success.
    """
    init("hccl")
    model_config = build_model_config()
    # layer config
    rank_id = get_rank()

    device_num = get_group_size()
    device_num_per_stage = device_num // model_config.num_stages
    stage_index = rank_id // device_num_per_stage

    per_batch_size = 2
    seq_length = 32
    ms.set_seed(0)

    micro_batch_num = 2

    model_config.stage_idx = stage_index

    rank_ids = [0, 1, 2, 3]
    pp_group = "pp_group"
    create_group(pp_group, rank_ids)

    send_info_list, recv_info_list = build_send_recv_info(stage_index, model_config, per_batch_size,
                                                          seq_length)

    # parameter
    pipeline_stage = build_pipeline_stage(model_config, pp_group, send_info_list, recv_info_list)

    input_shape = (per_batch_size * micro_batch_num, seq_length)
    input_ids = Tensor(np.ones(input_shape), dtype=ms.int32)  # (b, s)
    targets = Tensor(np.ones(input_shape), dtype=ms.int32)

    schedule = Schedule1F1B(pipeline_stage, micro_batch_num)
    if stage_index == 0:
        schedule.run(input_ids)
    elif stage_index == model_config.num_stages - 1:
        outputs = schedule.run()
        loss = calculate_loss(outputs, targets, micro_batch_num)
        assert np.allclose([loss], [5.386], atol=1e-3)
    else:
        schedule.run()


def test_schedule_1f1b():
    """
    Feature: Pipeline schedule 1f1b
    Description: Execute micro batches according to schedule
    Expectation: Run success.
    """
    init("hccl")
    model_config = build_model_config()
    # layer config
    rank_id = get_rank()

    device_num = get_group_size()
    device_num_per_stage = device_num // model_config.num_stages
    stage_index = rank_id // device_num_per_stage

    per_batch_size = 2
    seq_length = 32
    ms.set_seed(0)

    micro_batch_num = 8

    model_config.stage_idx = stage_index

    rank_ids = [0, 1, 2, 3]
    pp_group = "pp_group"
    create_group(pp_group, rank_ids)

    send_info_list, recv_info_list = build_send_recv_info(stage_index, model_config, per_batch_size,
                                                          seq_length)

    # parameter
    pipeline_stage = build_pipeline_stage(model_config, pp_group, send_info_list, recv_info_list)

    input_shape = (per_batch_size * micro_batch_num, seq_length)
    input_ids = Tensor(np.ones(input_shape), dtype=ms.int32)  # (b, s)
    targets = Tensor(np.ones(input_shape), dtype=ms.int32)

    schedule = Schedule1F1B(pipeline_stage, micro_batch_num)
    if stage_index == 0:
        schedule.run(input_ids)
    elif stage_index == model_config.num_stages - 1:
        outputs = schedule.run()
        loss = calculate_loss(outputs, targets, micro_batch_num)
        assert np.allclose([loss], [5.383], atol=1e-3)
    else:
        schedule.run()
