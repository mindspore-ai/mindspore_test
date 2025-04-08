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
import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.nn import Cell
from mindspore.ops import operations as P
from mindspore import Tensor, Parameter
from mindspore.communication import get_rank, init
from mindspore.parallel.shard import Layout

hidden_size, intermediate_size = 4, 64
S, E = 4, 4

def my_infer_dtype(*args):
    return args[0]

def my_infer_shape(*args):
    return args[0]

def my_hook(x, expert_id, counter):
    ep = 2
    local_counter = ops.AlltoAll(split_count=ep, split_dim=-1, concat_dim=-2)(counter)
    sl = ops.cast(counter.reshape(ep, -1).sum(dim=-1, keepdim=False), ms.int64)
    rl = ops.cast(local_counter.reshape(ep, -1).sum(dim=-1, keepdim=False), ms.int64)

    sl_x = sl * hidden_size
    rl_x = rl * hidden_size

    sl = ops.MoveTo()(sl, "CPU", True)
    rl = ops.MoveTo()(rl, "CPU", True)

    sl_x = ops.MoveTo()(sl_x, "CPU", True)
    rl_x = ops.MoveTo()(rl_x, "CPU", True)

    # 1.AllToAllV
    x = ops.AlltoAllV()(x.reshape(-1), sl_x, rl_x).reshape(1, -1, hidden_size)
    expert_id = ops.AlltoAllV()(expert_id.reshape(-1), sl, rl).reshape(1, -1)

    # 2.Resort
    _, sort_map = ops.sort(expert_id)
    _, unsort_map = ops.sort(ops.cast(sort_map, ms.float32))
    x = ops.gather(x, sort_map, axis=1, batch_dims=1)

    # 3.Set y
    y = x

    # 4.Unresort
    y = ops.gather(y, unsort_map, axis=1, batch_dims=1)

    # 5.AllToAllV
    y = ops.AlltoAllV()(y.reshape(-1), sl_x, rl_x).reshape(1, -1, hidden_size)
    return y

class FFN(nn.Cell):
    def __init__(self):
        super(FFN, self).__init__()
        self.rank_id = get_rank()
        self.dp = 2
        self.ep = 2

        # local expert indices:
        self.num_local_experts = E // self.ep
        self.bias = Parameter(Tensor(np.random.rand(1, 1, hidden_size), ms.float32), name="bias")

        self.outer_dp = self.dp // self.ep
        self.inner_dp = self.ep
        self.layout = Layout((self.outer_dp, self.inner_dp, 1, 1, 1), ("outer_dp", "inner_dp", "sp", "mp0", "mp1"))

        self.hook_ffn_forward = P.Morph(my_hook, my_infer_shape,
                                        my_infer_dtype).add_prim_attr("self_define_shard", True)
        self.hook_ffn_forward.shard(
            in_strategy=(
                self.layout(("outer_dp", "inner_dp"), "sp", "mp0"),
                self.layout(("outer_dp", "inner_dp"), "sp", "mp0"),
                self.layout(("outer_dp", "inner_dp"), "sp"),
                # self.layout("dp", "mp0")
            ),
            out_strategy=(
                self.layout(("outer_dp", "inner_dp"), "sp", "mp0"),
            )
        )

    def construct(self, ffn_x, ffn_expert_id, ffn_counter):
        ffn_x = self.hook_ffn_forward(ffn_x, ffn_expert_id, ffn_counter)
        return ffn_x

class TrainOneStep(Cell):
    def __init__(self, train_network, train_optimizer):
        super().__init__()
        self.network = train_network
        self.optimizer = train_optimizer

    def construct(self, train_x, train_expert_id, train_counter):
        train_loss = self.network(train_x, train_expert_id, train_counter)
        return train_loss

def init_env():
    init()
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', jit_level="O1")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

def load_value():
    rank_id = get_rank()
    if rank_id == 0:
        load_x = ms.Tensor(np.random.rand(1, S, hidden_size), dtype=ms.float32)
        load_expert_id = ms.Tensor([[[0], [1], [2], [3]]], dtype=ms.float32)
        load_counter = ms.Tensor([[1, 1, 1, 1]], ms.float32)
    else:
        load_x = ms.Tensor(np.random.rand(1, S, hidden_size), dtype=ms.float32)
        load_expert_id = ms.Tensor([[[0], [1], [2], [3]]], dtype=ms.float32)
        load_counter = ms.Tensor([[1, 1, 1, 1]], ms.float32)

    return load_x, load_expert_id, load_counter


if __name__ == '__main__':
    init_env()
    in_x, in_expert_id, in_counter = load_value()
    network = FFN()
    optimizer = nn.Momentum(filter(lambda x: x.requires_grad, network.get_parameters()), learning_rate=0.1,
                            momentum=0.9)

    trainonestep = TrainOneStep(network, optimizer)
    loss = trainonestep(in_x, in_expert_id, in_counter)
    print(f"run success, the loss is {loss}")
