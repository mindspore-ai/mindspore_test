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
import json
import pytest
import os
import subprocess
import shutil
import numpy as np

import mindspore as ms
from mindspore import context, Tensor
from mindspore.nn import Cell
import mindspore.ops as ops
from tests.ut.python.parallel.test_moe_net import MoEFFNet, Linear, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def check_output(dir_name, num_comm_ops=10):
    file = "%s/*validate*.ir" % dir_name
    prim_name = "Depend("
    tag_name = "split_concat_depend"
    output = subprocess.check_output(
        ["grep -r '%s' %s | grep '%s' |wc -l" % (prim_name, file, tag_name)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out > str(num_comm_ops)

class SplitConcatNet(Cell):
    def __init__(self, hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, sp, split_count, flag):
        super(SplitConcatNet, self).__init__()
        self.embed = Linear(in_channels=hidden_size, out_channels=hidden_size)
        self.head = Linear(in_channels=hidden_size, out_channels=hidden_size)
        self.moe_net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, sp)
        self.split = ops.Split(axis=2, output_num=split_count)
        self.concat = ops.Concat(2)
        self.split.shard(((dp, 1, 1, 1),))
        self.concat.shard(tuple((dp, 1, 1, 1) for _ in range(split_count)))
        self.split.add_prim_attr("enable_interleave", flag)
        self.concat.add_prim_attr("enable_interleave", flag)

    def construct(self, x):
        x = self.embed(x)
        output_list = []
        for sub_x in self.split(x):
            sub_output = self.moe_net(sub_x)
            output_list.append(sub_output)
        output = self.concat(output_list)
        output = self.head(output)
        return output

@pytest.mark.parametrize('split_count', [2, 8])
@pytest.mark.parametrize('parallel_flag', [1, 2])
def test_interleave_split_concat_branch(split_count, parallel_flag):
    """
    Feature: interleave split/concat branches.
    Description: interleave split/concat branches.
    Expectation: compile done without error and find split/concat branch control depend.
    """
    dir_name = "./split_concat_branch_interleave_" + str(split_count) + "_" + str(parallel_flag)
    config = {"enable_interleave_split_concat_branch": True,}
    with open("./parallel_speed_up_interleave.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True,
                        save_graphs_path=dir_name,
                        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_interleave.json"})

    context.set_context()
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=128,
                                      global_rank=0,
                                      enable_alltoall=True)
    hidden_size = 128
    ffn_hidden_size = 4 * hidden_size
    channel = 128
    expert_num = 16
    dp = 1
    ep = 16
    mp = 8
    sp = False
    net = SplitConcatNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, sp, split_count, parallel_flag)
    x = Tensor(np.ones([expert_num, expert_num, channel, hidden_size]), dtype=ms.float16)

    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    compile_net(net, x)
    check_output(dir_name)

    context.set_context(save_graphs=False)
    config = {"enable_interleave_split_concat_branch": False,}
    with open("./parallel_speed_up_interleave.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(
        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_interleave.json"})
