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

import os
import numpy as np
import json
from mindspore import context, Tensor
from mindspore.nn import Cell
from mindspore.common import dtype as mstype
from mindspore.parallel._transformer.moe import MoE
from mindspore.parallel._transformer import TransformerOpParallelConfig, MoEConfig
from parallel.utils.utils import compile_net, ParallelValidator


EXPERT_NUM = 64
CAPACITY_FACTOR = 8.0
AUX_LOSS_FACTOR = 0.01

DATA_PARALLEL = 8
MODEL_PARALLEL = 1
EXPERT_PARALLEL = 8

HIDDEN_SIZE = 2048
FFN_HIDDEN_SIZE = 2048 * 4


class Net(Cell):

    def __init__(self, hidden_size, ffn_hidden_size, moe_config, parallel_config):
        super(Net, self).__init__()
        self.output = MoE(hidden_size=hidden_size,
                          dropout_rate=0.1,
                          ffn_hidden_size=ffn_hidden_size,
                          param_init_type=mstype.float16,
                          hidden_act="fast_gelu",
                          moe_config=moe_config,
                          parallel_config=parallel_config.moe_parallel_config)

    def construct(self, x):
        mlp_logit, aux_loss = self.output(x)
        return mlp_logit, aux_loss


def test_single_rank_multi_expers_compile():
    """
    Feature: test compile MoE net which applies single rank multi experts
    Description: set device 8 with 64 experts
    Expectation: compile success
    """

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=8,
                                      global_rank=0,
                                      full_batch=True,
                                      enable_alltoall=True)

    moe_config = MoEConfig(
        expert_num=EXPERT_NUM,
        capacity_factor=CAPACITY_FACTOR,
        aux_loss_factor=AUX_LOSS_FACTOR,
    )

    context.set_context(jit_config={"jit_level": "O0"})
    if os.path.exists("./speed_up.json"):
        os.remove("./speed_up.json")
    a = {"enable_offloading_packed_experts": True}
    f = open("./speed_up.json", "w")
    f.write(json.dumps(a))
    f.close()
    context.set_context(ascend_config={"parallel_speed_up_json_path": "speed_up.json"})

    parallel_config = TransformerOpParallelConfig(data_parallel=DATA_PARALLEL,
                                                  model_parallel=MODEL_PARALLEL,
                                                  expert_parallel=EXPERT_PARALLEL)

    net = Net(HIDDEN_SIZE, FFN_HIDDEN_SIZE, moe_config, parallel_config)
    net.output.ffn.projection.matmul.add_prim_attr("expert_num", EXPERT_NUM)
    net.output.ffn.projection.matmul.add_prim_attr("pe_num", EXPERT_NUM // EXPERT_PARALLEL)
    data = Tensor(np.random.randn(1024, HIDDEN_SIZE), dtype=mstype.float16)
    phase = compile_net(net, data)
    validator = ParallelValidator(net, phase)
    assert validator.check_parameter_shape('x', [1024, 2048])
    assert validator.check_parameter_shape('output.ffn.projection.weight', [8, 8192, 2048])
