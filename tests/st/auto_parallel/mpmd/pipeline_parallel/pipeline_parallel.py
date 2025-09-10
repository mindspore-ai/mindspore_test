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
import copy
import numpy as np
from dataclasses import dataclass

import mindspore as ms
import mindspore.ops.functional as F

from mindspore import nn, Tensor, mint, ops
from mindspore.common import dtype as mstype
from mindspore.communication.management import init, get_rank, get_group_size, create_group
from mindspore.mint.nn import Linear, SiLU, LayerNorm, Embedding, CrossEntropyLoss
from mindspore.parallel import Layout
from mindspore.parallel.mpmd.pipeline_parallel import Schedule1F1B, PipelineStage, P2PInfo
from mindspore.parallel.spmd.hsdp.hsdp import hsdp


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


@dataclass
class RunningConfig:
    per_batch_size: int = 1
    seq_length: int = 32
    micro_batch_num: int = 1
    epoch_num: int = 1
    learning_rate: float = 0.001


class LinearWrapper:
    def __init__(self, cls, seed=0):
        self.cls = cls
        self.seed = seed

    def __call__(self, *args, **kwargs):
        ms.set_seed(self.seed)
        return self.cls(*args, **kwargs)


LinearWithSeed = LinearWrapper(Linear, seed=42)


class MockMLP(nn.Cell):
    def __init__(self, hidden_size, intermediate_size, compute_dtype=mstype.float16):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.w1 = LinearWithSeed(hidden_size, intermediate_size, dtype=compute_dtype)
        self.w2 = LinearWithSeed(intermediate_size, hidden_size, dtype=compute_dtype)
        self.w3 = LinearWithSeed(hidden_size, intermediate_size, dtype=compute_dtype)
        self.silu = SiLU(inplace=False)

    def construct(self, x):
        gate = self.silu(self.w1(x))
        hidden = self.w3(x)
        hidden = mint.mul(gate, hidden)
        out = self.w2(hidden)
        return out


class TransformerBlock(nn.Cell):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, compyte_dtype):
        super().__init__()
        self.ln1 = LayerNorm((hidden_size,))
        self.attn = MockAttention(num_attention_heads, intermediate_size, compyte_dtype)
        self.ln2 = LayerNorm((hidden_size,))
        self.mlp = MockMLP(hidden_size, intermediate_size, compyte_dtype)

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
        ms.set_seed(42)
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
                                                          self.num_attention_heads, config.compute_dtype)
        self.ln_f = nn.LayerNorm((self.hidden_size,)) if config.stage_idx == config.num_stages - 1 else None

    def construct(self, input_ids, mask=None):
        x = self.tok_embedding(input_ids) if self.tok_embedding is not None else input_ids
        x = F.cast(x, self.compute_dtype)
        for layer in self.layers.values():
            x = layer(x, mask)
        if self.ln_f is None:
            return x
        norm_output = self.ln_f(x)
        return F.cast(norm_output, self.compute_dtype)


class MockLM(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = MockModel(config)
        self.lm_head = LinearWithSeed(config.hidden_size, config.vocab_size,
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

        self.q_proj = LinearWithSeed(hidden_size, hidden_size, dtype=compute_dtype)
        self.k_proj = LinearWithSeed(hidden_size, hidden_size, dtype=compute_dtype)
        self.v_proj = LinearWithSeed(hidden_size, hidden_size, dtype=compute_dtype)
        self.out_proj = LinearWithSeed(hidden_size, hidden_size, dtype=compute_dtype)

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
    model_config.compute_dtype = mstype.float16
    return model_config


def build_send_recv_info(stage_index, model_config, per_batch_size, seq_length):
    recv_info_list = []
    send_info_list = []
    if stage_index == 0:
        send_info = P2PInfo(shape=[per_batch_size, seq_length, model_config.hidden_size],
                            dtype=model_config.compute_dtype)
        send_info_list.append(send_info)
    elif stage_index == model_config.num_stages - 1:
        # for forward recv
        recv_info = P2PInfo(shape=[per_batch_size, seq_length, model_config.hidden_size],
                            dtype=model_config.compute_dtype)
        recv_info_list.append(recv_info)
    else:
        # for forward recv
        recv_info = P2PInfo(shape=[per_batch_size, seq_length, model_config.hidden_size],
                            dtype=model_config.compute_dtype)
        recv_info_list.append(recv_info)
        # for forward send
        send_info = P2PInfo(shape=[per_batch_size, seq_length, model_config.hidden_size],
                            dtype=model_config.compute_dtype)
        send_info_list.append(send_info)
    return send_info_list, recv_info_list


def build_pipeline_stage(model_config, pp_group, send_info_list, recv_info_list, micro_batch_num=1, shard_size=1,
                         w_layout=None, norm_layout=None):
    stage_kwargs = {
        'stage_num': model_config.num_stages,
        'send_info': send_info_list,
        'recv_info': recv_info_list,
        'group': pp_group
    }
    model = MockLM(model_config)
    # set param layout
    for _, param in model.parameters_and_names():
        if len(param.shape) > 1:
            param = param.local_to_global(w_layout)
        else:
            param = param.local_to_global(norm_layout)
    # enable hsdp
    hsdp(model, shard_size=shard_size, enable_grad_accumulation=True)
    # build pipeline_stage
    pipeline_stage = PipelineStage(model, model_config.stage_idx, **stage_kwargs)
    return pipeline_stage


def loss_fn(outputs, targets, vocab_size):
    loss_func = CrossEntropyLoss()
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    outputs = mint.reshape(outputs, (-1, vocab_size))
    targets = mint.reshape(targets, (-1,))
    return loss_func(outputs, targets)


# loss_fn
def calculate_loss(model_outputs, model_targets, micro_batch_num, vocab_size=256):
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
    running_config = RunningConfig()
    running_config.per_batch_size = 2
    running_config.micro_batch_num = 2
    running_config.epoch_num = 5

    model_config = build_model_config()
    model_config.compute_dtype = mstype.float32

    init("hccl")

    input_id_list = []
    target_list = []
    input_shape = (running_config.per_batch_size * running_config.micro_batch_num, running_config.seq_length)

    for epoch_idx in range(running_config.epoch_num):
        np.random.seed(42 + epoch_idx)
        input_id_list.append(Tensor(np.random.randint(0, model_config.vocab_size, input_shape), dtype=mstype.int32))
        target_list.append(Tensor(np.random.randint(0, model_config.vocab_size, input_shape), dtype=mstype.int32))

    # for standalone
    standalone_loss = run_standalone(input_id_list, target_list, running_config, copy.copy(model_config))
    # for parallel
    parallel_model_config = copy.copy(model_config)
    layout_rank_list, group_rank_list = build_rank_list(parallel_model_config)
    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"), rank_list=layout_rank_list)
    x_layout = layout("dp", "mp")
    w_layout = layout("mp", "None")
    norm_layout = layout("None")
    parallel_loss, parallel_model_config = run_parallel(input_id_list, target_list, running_config,
                                                        parallel_model_config, x_layout, w_layout, norm_layout,
                                                        rank_list=group_rank_list)
    if parallel_model_config.stage_idx == parallel_model_config.num_stages - 1:
        assert np.allclose(standalone_loss, parallel_loss, 1e-3, 1e-3)


def run_standalone(input_id_list: list, target_list: list, run_config: RunningConfig, model_config: ModelArgs):
    model_config.num_stages = 1
    model_config.stage_idx = 0

    model = MockLM(model_config)
    model.set_grad(True)

    optimizer = nn.Adam(model.trainable_params(), learning_rate=run_config.learning_rate)
    loss_list = []
    for epoch_idx in range(len(input_id_list)):
        input_ids = input_id_list[epoch_idx]
        target = target_list[epoch_idx]
        outputs = model(input_ids)
        backward_func = ops.GradOperation(get_by_list=True, get_all=True, sens_param=False)(model,
                                                                                            model.trainable_params())
        grads = backward_func(input_ids)
        optimizer(grads[1])
        loss = loss_fn(outputs, target, vocab_size=model_config.vocab_size)
        print(f"[standalone], epoch: {epoch_idx + 1}/{run_config.epoch_num}, loss is: {loss}")
        loss_list.append(loss)
    return loss_list


def run_parallel(input_id_list, target_list, run_config: RunningConfig, model_config: ModelArgs, x_layout, w_layout,
                 norm_layout, rank_list, dynamic_shape=False, dynamic_rank=False):
    pp_group = "pp_group"
    create_group(pp_group, rank_list)
    if dynamic_shape:
        send_info_list, recv_info_list = build_send_recv_info_dynamic_shape(model_config.stage_idx, model_config,
                                                                            run_config.seq_length)
    elif dynamic_rank:
        send_info_list, recv_info_list = build_send_recv_info_dynamic_rank(model_config.stage_idx, model_config)
    else:
        send_info_list, recv_info_list = build_send_recv_info(model_config.stage_idx, model_config,
                                                              run_config.per_batch_size,
                                                              run_config.seq_length)

    # parameter
    pipeline_stage = build_pipeline_stage(model_config, pp_group, send_info_list, recv_info_list,
                                          micro_batch_num=run_config.micro_batch_num, w_layout=w_layout,
                                          norm_layout=norm_layout)
    schedule = Schedule1F1B(pipeline_stage, run_config.micro_batch_num)
    optimizer = nn.Adam(pipeline_stage.submodule.trainable_params(), learning_rate=run_config.learning_rate)
    loss_list = []
    for epoch_idx in range(len(input_id_list)):
        input_ids = input_id_list[epoch_idx].local_to_global(x_layout)
        target = target_list[epoch_idx]
        pipeline_stage.submodule.zero_grads()
        pipeline_stage.submodule.set_requires_grad_sync(False)
        if model_config.stage_idx == 0:
            _, grads = schedule.run(input_ids)
            optimizer(grads)
        elif model_config.stage_idx == model_config.num_stages - 1:
            outputs, grads = schedule.run()
            loss = calculate_loss(outputs, target, run_config.micro_batch_num)
            print(f"[parallel], epoch: {epoch_idx + 1}/{run_config.epoch_num}, loss is: {loss}")
            loss_list.append(loss)
            optimizer(grads)
        else:
            _, grads = schedule.run()
            optimizer(grads)
    return loss_list, model_config


def test_schedule_1f1b_precision():
    """
    Feature: Pipeline schedule 1f1b
    Description: Execute micro batches according to schedule and validate precision
    Expectation: Run success.
    """
    running_config = RunningConfig()
    running_config.per_batch_size = 2
    running_config.micro_batch_num = 4
    running_config.epoch_num = 5

    model_config = build_model_config()
    model_config.compute_dtype = mstype.float32
    input_id_list = []
    target_list = []
    input_shape = (running_config.per_batch_size * running_config.micro_batch_num, running_config.seq_length)
    for epoch_idx in range(running_config.epoch_num):
        np.random.seed(42 + epoch_idx)
        input_id_list.append(Tensor(np.random.randint(0, model_config.vocab_size, input_shape), dtype=mstype.int32))
        target_list.append(Tensor(np.random.randint(0, model_config.vocab_size, input_shape), dtype=mstype.int32))

    init("hccl")
    standalone_loss = run_standalone(input_id_list, target_list, running_config, copy.copy(model_config))
    # parallel
    parallel_model_config = copy.copy(model_config)
    layout_rank_list, group_rank_list = build_rank_list(parallel_model_config)
    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"), rank_list=layout_rank_list)
    x_layout = layout("dp", "mp")
    w_layout = layout("mp", "None")
    norm_layout = layout("None")
    parallel_loss, parallel_model_config = run_parallel(input_id_list, target_list, running_config,
                                                        parallel_model_config, x_layout, w_layout, norm_layout,
                                                        rank_list=group_rank_list)
    if parallel_model_config.stage_idx == parallel_model_config.num_stages - 1:
        assert np.allclose(standalone_loss, parallel_loss, 1e-3, 1e-3)


def build_rank_list(model_config):
    # build rank list
    rank_id = get_rank()
    device_num = get_group_size()
    device_num_per_stage = device_num // model_config.num_stages
    stage_index = rank_id // device_num_per_stage
    model_config.stage_idx = stage_index
    layout_rank_list = [device_num_per_stage * stage_index + i for i in range(device_num_per_stage)]
    local_stage_rank_id = rank_id % device_num_per_stage
    group_rank_list = [local_stage_rank_id + i * device_num_per_stage for i in range(model_config.num_stages)]
    return layout_rank_list, group_rank_list


def build_send_recv_info_dynamic_shape(stage_index, model_config, seq_length):
    recv_info_list = []
    send_info_list = []
    if stage_index == 0:
        send_info = P2PInfo(shape=[-1, seq_length, model_config.hidden_size], dtype=model_config.compute_dtype,
                            dyn_shape=True)
        send_info_list.append(send_info)
    elif stage_index == model_config.num_stages - 1:
        # for forward recv
        recv_info = P2PInfo(shape=[-1, seq_length, model_config.hidden_size], dtype=model_config.compute_dtype,
                            dyn_shape=True)
        recv_info_list.append(recv_info)
    else:
        # for forward recv
        recv_info = P2PInfo(shape=[-1, seq_length, model_config.hidden_size], dtype=model_config.compute_dtype,
                            dyn_shape=True)
        recv_info_list.append(recv_info)
        # for forward send
        send_info = P2PInfo(shape=[-1, seq_length, model_config.hidden_size], dtype=model_config.compute_dtype,
                            dyn_shape=True)
        send_info_list.append(send_info)
    return send_info_list, recv_info_list


def test_pipeline_dynamic_shape_precision():
    """
    Feature: Pipeline schedule 1f1b + dynamic shape
    Description: Execute micro batches according to schedule
    Expectation: Run success.
    """
    running_config = RunningConfig()
    running_config.per_batch_size = 2
    running_config.micro_batch_num = 4
    running_config.epoch_num = 5

    model_config = build_model_config()
    model_config.compute_dtype = mstype.float32
    input_id_list = []
    target_list = []
    input_shape = (running_config.per_batch_size * running_config.micro_batch_num, running_config.seq_length)
    for epoch_idx in range(running_config.epoch_num):
        np.random.seed(42 + epoch_idx)
        input_id_list.append(Tensor(np.random.randint(0, model_config.vocab_size, input_shape), dtype=mstype.int32))
        target_list.append(Tensor(np.random.randint(0, model_config.vocab_size, input_shape), dtype=mstype.int32))

    init("hccl")
    standalone_loss = run_standalone(input_id_list, target_list, running_config, copy.copy(model_config))
    # parallel
    parallel_model_config = copy.copy(model_config)
    layout_rank_list, group_rank_list = build_rank_list(parallel_model_config)
    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"), rank_list=layout_rank_list)
    x_layout = layout("dp", "mp")
    w_layout = layout("mp", "None")
    norm_layout = layout("None")
    parallel_loss, parallel_model_config = run_parallel(input_id_list, target_list, running_config,
                                                        parallel_model_config, x_layout, w_layout, norm_layout,
                                                        rank_list=group_rank_list, dynamic_shape=True)
    if parallel_model_config.stage_idx == parallel_model_config.num_stages - 1:
        assert np.allclose(standalone_loss, parallel_loss, 1e-3, 1e-3)


def build_send_recv_info_dynamic_rank(stage_index, model_config):
    recv_info_list = []
    send_info_list = []
    if stage_index == 0:
        send_info = P2PInfo(shape=None, dtype=model_config.compute_dtype, dyn_rank=True)
        send_info_list.append(send_info)
    elif stage_index == model_config.num_stages - 1:
        # for forward recv
        recv_info = P2PInfo(shape=None, dtype=model_config.compute_dtype, dyn_rank=True)
        recv_info_list.append(recv_info)
    else:
        # for forward recv
        recv_info = P2PInfo(shape=None, dtype=model_config.compute_dtype, dyn_rank=True)
        recv_info_list.append(recv_info)
        # for forward send
        send_info = P2PInfo(shape=None, dtype=model_config.compute_dtype, dyn_rank=True)
        send_info_list.append(send_info)
    return send_info_list, recv_info_list


def test_pipeline_dynamic_rank_precision():
    """
    Feature: Pipeline schedule 1f1b + dynamic rank
    Description: Execute micro batches according to schedule
    Expectation: Run success.
    """
    running_config = RunningConfig()
    running_config.per_batch_size = 2
    running_config.micro_batch_num = 4
    running_config.epoch_num = 5

    model_config = build_model_config()
    model_config.compute_dtype = mstype.float32
    input_id_list = []
    target_list = []
    input_shape = (running_config.per_batch_size * running_config.micro_batch_num, running_config.seq_length)
    for epoch_idx in range(running_config.epoch_num):
        np.random.seed(42 + epoch_idx)
        input_id_list.append(Tensor(np.random.randint(0, model_config.vocab_size, input_shape), dtype=mstype.int32))
        target_list.append(Tensor(np.random.randint(0, model_config.vocab_size, input_shape), dtype=mstype.int32))

    init("hccl")
    standalone_loss = run_standalone(input_id_list, target_list, running_config, copy.copy(model_config))
    # parallel
    parallel_model_config = copy.copy(model_config)
    layout_rank_list, group_rank_list = build_rank_list(parallel_model_config)
    dp = 1
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"), rank_list=layout_rank_list)
    x_layout = layout("dp", "mp")
    w_layout = layout("mp", "None")
    norm_layout = layout("None")
    parallel_loss, parallel_model_config = run_parallel(input_id_list, target_list, running_config,
                                                        parallel_model_config, x_layout, w_layout, norm_layout,
                                                        rank_list=group_rank_list,
                                                        dynamic_rank=True)
    if parallel_model_config.stage_idx == parallel_model_config.num_stages - 1:
        assert np.allclose(standalone_loss, parallel_loss, 1e-3, 1e-3)
