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
import numpy as np

from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore.ops.auto_generate import MoeComputeExpertTokens
from parallel.utils.utils import compile_net

class MoeComputeExpertTokensNet(Cell):
    def __init__(self, strategy):
        super().__init__()
        self.net = MoeComputeExpertTokens().shard(strategy)

    def construct(self, sorted_experts, num_experts):
        out = self.net(sorted_experts, num_experts)
        return out

def generate_random_input(sorted_expert_len, num_experts):
    random_int_list = []
    for _ in range(sorted_expert_len):
        random_int_list.append(np.random.randint(0, num_experts))
    sorted_experts = np.sort(random_int_list).astype(np.int32)
    return sorted_experts


def test_moe_compute_expert_tokens_case0():
    """
    Feature: Test moe_compute_expert_tokens auto parallel
    Description: semi_auto_parallel
    Expectation: compile done without error.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=1, global_rank=0)
    context.set_context(device_target="Ascend")

    sorted_expert_len = 1024
    num_experts = 32
    sorted_experts = generate_random_input(sorted_expert_len, num_experts)
    strategy = ((1,),)
    net = MoeComputeExpertTokensNet(strategy)
    compile_net(net, Tensor(sorted_experts), num_experts)
