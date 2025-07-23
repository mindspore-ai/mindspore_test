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
from tests.ut.python.parallel.test_moe_net import MoEFFNet, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def check_output(dir_name, num_comm_ops=1):
    file = "%s/*validate*.ir" % dir_name
    prim_name = "ReduceScatter("
    tag_name = "forward_op"
    output = subprocess.check_output(
        ["grep -r '%s' %s | grep '%s' |wc -l" % (prim_name, file, tag_name)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out >= str(num_comm_ops)


def set_test_context(dir_name):
    config = {"enable_allreduce_slice_to_reducescatter": True,}
    with open("./parallel_speed_up_test.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True,
                        save_graphs_path=dir_name,
                        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_test.json"})

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=128,
                                      global_rank=0,
                                      enable_alltoall=True)


def compile_add_check_output(dir_name, net, x):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

    compile_net(net, x)
    check_output(dir_name)

    context.set_context(save_graphs=False)
    config = {"enable_allreduce_slice_to_reducescatter": False,}
    with open("./parallel_speed_up_test.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(
        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_test.json"})


@pytest.mark.parametrize('has_bias', [False, True])
def test_batch_matmul_opt(has_bias):
    """
    Feature: BatchMatMul+allreduce+split to BatchMatMul+reducescatter.
    Description: BatchMatMul+allreduce+split to BatchMatMul+reducescatter.
    Expectation: compile done without error.
    """
    dir_name = "./test_batch_matmul_opt"
    if has_bias:
        dir_name = dir_name + "_bias_true"
    set_test_context(dir_name)

    hidden_size = 64
    ffn_hidden_size = 4 * hidden_size
    channel = 2256
    expert_num = 16
    dp = 1
    ep = 16
    mp = 8
    sp = False
    transpose_b = True
    net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, sp, has_bias, transpose_b)
    x = Tensor(np.ones([expert_num, expert_num, channel, hidden_size]), dtype=ms.float16)

    compile_add_check_output(dir_name, net, x)


@pytest.mark.parametrize('has_bias', [False, True])
def test_batch_matmul_opt_with_mp_larger_than_ep(has_bias):
    """
    Feature: BatchMatMul+allreduce+split to BatchMatMul+reducescatter.
    Description: BatchMatMul+allreduce+split to BatchMatMul+reducescatter with mp > ep.
    Expectation: compile done without error.
    """
    dir_name = "./test_batch_matmul_opt_with_mp_larger_than_ep"
    if has_bias:
        dir_name = dir_name + "_bias_true"
    set_test_context(dir_name)

    hidden_size = 64
    ffn_hidden_size = 4 * hidden_size
    channel = 2256
    expert_num = 16
    dp = 1
    ep = 8
    mp = 16
    sp = False
    net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, sp, has_bias)
    x = Tensor(np.ones([expert_num, expert_num, channel, hidden_size]), dtype=ms.float16)

    compile_add_check_output(dir_name, net, x)


@pytest.mark.parametrize('has_bias', [False, True])
def test_batch_matmul_opt_with_outer_dp(has_bias):
    """
    Feature: BatchMatMul+allreduce+split to BatchMatMul+reducescatter.
    Description: BatchMatMul+allreduce+split to BatchMatMul+reducescatter with outer dp > 1.
    Expectation: compile done without error.
    """
    dir_name = "./test_batch_matmul_opt_with_outer_dp"
    if has_bias:
        dir_name = dir_name + "_bias_true"
    set_test_context(dir_name)

    hidden_size = 64
    ffn_hidden_size = 4 * hidden_size
    channel = 2256
    expert_num = 16
    dp = 2
    ep = 8
    mp = 8
    sp = False
    net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, sp, has_bias)
    x = Tensor(np.ones([dp, ep, expert_num, channel, hidden_size]), dtype=ms.float16)

    compile_add_check_output(dir_name, net, x)


@pytest.mark.parametrize('has_bias', [False, True])
def test_batch_matmul_opt_with_sp(has_bias):
    """
    Feature: BatchMatMul+allreduce+split to BatchMatMul+reducescatter.
    Description: BatchMatMul+allreduce+split to BatchMatMul+reducescatter with sp = true.
    Expectation: compile done without error.
    """
    dir_name = "./test_batch_matmul_opt_with_sp"
    if has_bias:
        dir_name = dir_name + "_bias_true"
    set_test_context(dir_name)

    hidden_size = 64
    ffn_hidden_size = 4 * hidden_size
    channel = 2256
    expert_num = 16
    dp = 2
    ep = 8
    mp = 8
    sp = True
    net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, sp, has_bias)
    x = Tensor(np.ones([dp, ep, expert_num, channel, hidden_size]), dtype=ms.float16)

    compile_add_check_output(dir_name, net, x)

def test_batch_matmul_sharding_output_by_layout():
    """
    Feature: BatchMatMul sharding output.
    Description: BatchMatMul with output strategy and generate reduce_scatter.
    Expectation: compile done without error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=128,
                                      global_rank=0,
                                      enable_alltoall=True)
    hidden_size = 64
    ffn_hidden_size = 4 * hidden_size
    channel = 2256
    expert_num = 16
    dp = 2
    ep = 8
    mp = 8
    sp = True
    net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, sp, True, True, True)
    x = Tensor(np.ones([dp, ep, expert_num, channel, hidden_size]), dtype=ms.float16)
    compile_net(net, x)

def test_batch_matmul_sharding_output_by_stra():
    """
    Feature: BatchMatMul sharding output.
    Description: BatchMatMul with output strategy and generate reduce_scatter.
    Expectation: compile done without error.
    """
    context.set_context(mode=context.GRAPH_MODE)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=128,
                                      global_rank=0,
                                      enable_alltoall=True)
    hidden_size = 64
    ffn_hidden_size = 4 * hidden_size
    channel = 2256
    expert_num = 16
    dp = 2
    ep = 8
    mp = 8
    sp = True
    net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, sp, True, False, True)
    x = Tensor(np.ones([dp, ep, expert_num, channel, hidden_size]), dtype=ms.float16)
    compile_net(net, x)
