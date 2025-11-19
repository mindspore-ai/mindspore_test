"""
Copyright 2025 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pytest

import mindspore as ms
from mindspore import context
from mindspore import nn
from mindspore import ops
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.communication import get_rank, init
from mindspore.ops.auto_generate import (IndexSelect)
from mindspore.parallel.shard import Layout


def my_infer_dtype(*args):
    return args[0]


def my_infer_shape(*args):
    return args[0]


def no_depend_net(x, expert_ids, counter):
    ep = 2
    local_counter = ops.AlltoAll(split_count=ep, split_dim=-1, concat_dim=-2)(counter)
    sl = ops.cast(counter.reshape(ep, -1).sum(dim=-1, keepdim=False), ms.int64)
    rl = ops.cast(local_counter. reshape(ep, -1).sum(dim=-1, keepdim=False), ms.int64)

    sl = ops.MoveTo()(sl, "CPU", False)
    rl = ops.MoveTo()(rl, "CPU", False)

    x = x.reshape(-1, x.shape[-1])
    hidden_size = x.shape[1]
    expert_ids = expert_ids.reshape(-1)
    sorted_expert_ids, dispatch_idx = ops.sort(expert_ids.astype(ms.float32))
    sorted_expert_ids = sorted_expert_ids.astype(ms.int32)

    mask = ops.logical_and(sorted_expert_ids >= 0, sorted_expert_ids < 2)
    mask = mask.reshape(-1)

    idx = ops.NonZero()(mask.reshape(-1))
    idx = idx.reshape(-1)

    dispatch_idx = IndexSelect()(dispatch_idx, 0, idx)

    excombine_whiteboard = x * Tensor(0.0, dtype=ms.bfloat16)

    x = IndexSelect()(x, 0, dispatch_idx)
    x = ops.AlltoAllV(block_size=hidden_size)(x.reshape(-1), sl, rl).reshape(-1, hidden_size)
    x = ops.AlltoAllV(block_size=hidden_size)(x.reshape(-1), rl, sl).reshape(-1, hidden_size)

    x = excombine_whiteboard.index_add_(0, dispatch_idx.reshape(-1), x)
    x = ops.AllReduce()(x)
    y = x
    return y



def depend_net(x, expert_ids, counter):
    ep = 2
    local_counter = ops.AlltoAll(split_count=ep, split_dim=-1, concat_dim=-2)(counter)
    sl = ops.cast(counter.reshape(ep, -1).sum(dim=-1, keepdim=False), ms.int64)
    rl = ops.cast(local_counter. reshape(ep, -1).sum(dim=-1, keepdim=False), ms.int64)

    sl = ops.MoveTo()(sl, "CPU", False)
    rl = ops.MoveTo()(rl, "CPU", False)

    x = x.reshape(-1, x.shape[-1])
    hidden_size = x.shape[1]
    expert_ids = expert_ids.reshape(-1)
    sorted_expert_ids, dispatch_idx = ops.sort(expert_ids.astype(ms.float32))
    sorted_expert_ids = sorted_expert_ids.astype(ms.int32)

    mask = ops.logical_and(sorted_expert_ids >= 0, sorted_expert_ids < 2)
    mask = mask.reshape(-1)

    mask = ops.Depend()(mask, sl)
    mask = ops.Depend()(mask, rl)
    idx = ops.NonZero()(mask.reshape(-1))
    sl = ops.Depend()(sl, idx)
    rl = ops.Depend()(rl, idx)
    idx = idx.reshape(-1)

    dispatch_idx = IndexSelect()(dispatch_idx, 0, idx)

    excombine_whiteboard = x * Tensor(0.0, dtype=ms.bfloat16)

    x = IndexSelect()(x, 0, dispatch_idx)
    x = ops.AlltoAllV(block_size=hidden_size)(x.reshape(-1), sl, rl).reshape(-1, hidden_size)
    x = ops.AlltoAllV(block_size=hidden_size)(x.reshape(-1), rl, sl).reshape(-1, hidden_size)

    x = excombine_whiteboard.index_add_(0, dispatch_idx.reshape(-1), x)
    x = ops.AllReduce()(x)
    y = x
    return y

class FFN(nn.Cell):
    def __init__(self, fn):
        super().__init__()

        self.dp = 2
        self.layout = Layout((self.dp, 1, 1, 1), ("dp", "sp", "mp0", "mp1"))

        self.add = P.Add().shard(((self.dp, 1, 1), (1, 1, 1)))
        self.reduce_sum = P.ReduceSum().shard(((1, 1, 1), ))

        self.hook_ffn_forward = P.Morph(fn, my_infer_shape, my_infer_dtype).add_prim_attr("self_define_shard", True)
        self.hook_ffn_forward.shard(in_strategy=(self.layout("dp", "sp", "mp0"),
                                                 self.layout("dp", "sp"),
                                                 self.layout("dp", "sp")),
                                    out_strategy=(self.layout("dp", "sp", "mp0"),))

    def construct(self, x, expert_id, counter):
        x = self.hook_ffn_forward(x, expert_id, counter)
        loss = self.reduce_sum(x)
        return loss


def init_env():
    init()
    seed = 123
    context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", jit_level="O1")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_context(deterministic="ON")
    np.random.seed(seed)


def load_value():
    rank_id = get_rank()
    if rank_id == 0:
        data = np.arange(1, 11).reshape(1, -1, 1) * np.ones((1, 10, 4))
        x = Tensor(data, ms.float32)
        expert_id = ms.Tensor([[3, 2, 0, 1, 0, 2, 2, 1, 2, 1]], ms.float32)
        counter = ms.Tensor([[2, 3]], ms.float32)
    else:
        data = np.arange(11, 21).reshape(1, -1, 1) * np.ones((1, 10, 4))
        x = ms.Tensor(data, ms.float32)
        expert_id = ms.Tensor([[2, 0, 1, 2, 1, 2, 3, 1, 3, 3]], ms.float32)
        counter = ms.Tensor([[1, 3]], ms.float32)
    return x, expert_id, counter


def test_non_blocking_moveto_check_exec_order():
    """
    Feature: Async MoveTo for AlltoAllV sl and rl.
    Description: Check execution order for non-blocking MoveTo.
    Expectation: Raise Exception for wrong execution order.
    """
    init_env()
    x, expert_id, counter = load_value()
    network = FFN(no_depend_net)
    with pytest.raises(Exception, match="Validation for non-blocking MoveTo failed"):
        network(x, expert_id, counter)


def test_non_blocking_moveto_check_exec_order_with_depend():
    """
    Feature: Async MoveTo for AlltoAllV sl and rl.
    Description: Check execution order for non-blocking MoveTo.
    Expectation: Run successfully.
    """
    init_env()
    x, expert_id, counter = load_value()
    network = FFN(depend_net)
    network(x, expert_id, counter)
