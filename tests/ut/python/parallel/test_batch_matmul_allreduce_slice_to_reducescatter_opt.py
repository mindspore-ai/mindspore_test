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
from mindspore import context, Tensor, Parameter
from mindspore.common.api import _cell_graph_executor
from mindspore.common.initializer import initializer
from mindspore.nn import Cell, TrainOneStepCell, Momentum
import mindspore.common.dtype as mstype
import mindspore.ops as ops


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_context(mode=context.GRAPH_MODE, save_graphs=True, save_graphs_path="./batchmatmul_allreduce_opt")


class Linear(Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 transpose_b=True,
                 expert_num=1,
                 outer_batch=1,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        self.expert_num = expert_num
        self.outer_batch = outer_batch
        self.transpose_b = transpose_b
        self.expert_flag = True
        self.weight = Parameter(initializer(weight_init, [self.expert_num] + weight_shape, param_init_type),
                                name="weight")
        self.matmul = ops.BatchMatMul(transpose_b=transpose_b)

        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = Parameter(initializer(bias_init, [1, self.expert_num, 1, out_channels], param_init_type),
                                  name="bias")
            self.bias.parallel_optimizer = False
            self.bias_add = ops.Add()

        self.dtype = compute_dtype
        self.cast = ops.Cast()
        self.reshape = ops.Reshape()
        self.shape = ops.Shape()

    def construct(self, x):
        out_shape = self.shape(x)[:-1] + (self.out_channels,)
        x = self.reshape(x, (-1, self.in_channels))
        if self.expert_flag:
            x = self.reshape(x, (self.outer_batch, self.expert_num, -1, self.in_channels))
        ori_dtype = ops.dtype(x)
        weight = self.cast(self.weight, self.dtype)
        x = self.cast(x, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        x = ops.cast(x, ori_dtype)
        output = self.reshape(x, out_shape)
        return output

    def shard(self, strategy_matmul, strategy_bias=None, out_strategy_matmul=None):
        self.matmul.shard(in_strategy=strategy_matmul, out_strategy=out_strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        return self


class MoEFFNet(Cell):
    def __init__(self, hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, has_bias=True):
        super(MoEFFNet, self).__init__()
        input_size = hidden_size
        output_size = ffn_hidden_size
        param_init_type = mstype.float16
        compute_dtype = mstype.float16
        self.mapping = Linear(in_channels=input_size,
                              out_channels=output_size,
                              has_bias=has_bias,
                              transpose_b=False,
                              expert_num=expert_num,
                              outer_batch=dp,
                              param_init_type=param_init_type,
                              compute_dtype=compute_dtype)

        self.projection = Linear(in_channels=output_size,
                                 out_channels=input_size,
                                 has_bias=has_bias,
                                 transpose_b=False,
                                 expert_num=expert_num,
                                 outer_batch=dp,
                                 param_init_type=param_init_type,
                                 compute_dtype=compute_dtype)

        self.mapping.shard(strategy_matmul=((dp, ep, 1, 1), (ep, 1, mp)),
                           strategy_bias=((dp, ep, 1, mp), (1, ep, 1, mp)))
        self.projection.shard(strategy_matmul=((dp, ep, 1, mp), (ep, mp, 1)),
                              strategy_bias=((dp, ep, mp, 1), (1, ep, 1, 1)))
        self.shape = ops.Shape()
        self.reshape = ops.Reshape()
        self.stride_slice_ep = ops.StridedSlice().shard(((ep, 1, 1, 1),))
        self.stride_slice_ep_mp = ops.StridedSlice().shard(((ep, 1, mp, 1),))


    def construct(self, x):
        x_shape = self.shape(x)
        x = self.stride_slice_ep(x, (0, 0, 0, 0), x_shape, (1, 1, 1, 1))
        hidden = self.mapping(x)
        output = self.projection(hidden)
        output1 = self.reshape(output, x_shape)
        output2 = self.stride_slice_ep_mp(output1, (0, 0, 0, 0), x_shape, (1, 1, 1, 1))
        return output2



def compile_net(net, x):
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    train_net = TrainOneStepCell(net, optimizer)
    train_net.set_train()
    _cell_graph_executor.compile(train_net, x)
    context.reset_auto_parallel_context()


def check_output(num_comm_ops=1):
    file = "./batchmatmul_allreduce_opt/rank_0/*validate*.ir"
    prim_name = "ReduceScatter("
    tag_name = "forward_op"
    output = subprocess.check_output(
        ["grep -r '%s' %s | grep '%s' |wc -l" % (prim_name, file, tag_name)],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == str(num_comm_ops)


@pytest.mark.parametrize('has_bias', [False, True])
def test_batch_matmul_opt(has_bias):
    """
    Feature: BatchMatMul+allreduce+split to BatchMatMul+reducescatter.
    Description: BatchMatMul+allreduce+split to BatchMatMul+reducescatter.
    Expectation: compile done without error.
    """
    config = {"enable_allreduce_slice_to_reducescatter": True,}
    with open("./parallel_speed_up_test.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(
        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_test.json"})

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=128,
                                      global_rank=0,
                                      enable_alltoall=True)
    hidden_size = 4096
    ffn_hidden_size = 4 * hidden_size
    channel = 2256
    expert_num = 16
    dp = 1
    ep = 16
    mp = 8
    net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, has_bias)
    x = Tensor(np.ones([expert_num, expert_num, channel, hidden_size]), dtype=ms.float16)

    if os.path.exists("./batchmatmul_allreduce_opt/rank_0"):
        shutil.rmtree("./batchmatmul_allreduce_opt/rank_0")

    compile_net(net, x)
    check_output()

    context.set_context(save_graphs=False)
    config = {"enable_allreduce_slice_to_reducescatter": False,}
    with open("./parallel_speed_up_test.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(
        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_test.json"})


@pytest.mark.parametrize('has_bias', [False, True])
def test_batch_matmul_opt_with_mp_larger_than_ep(has_bias):
    """
    Feature: BatchMatMul+allreduce+split to BatchMatMul+reducescatter.
    Description: BatchMatMul+allreduce+split to BatchMatMul+reducescatter with mp > ep.
    Expectation: compile done without error.
    """
    config = {"enable_allreduce_slice_to_reducescatter": True,}
    with open("./parallel_speed_up_test.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(
        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_test.json"})

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel",
                                      device_num=128,
                                      global_rank=0,
                                      enable_alltoall=True)
    hidden_size = 4096
    ffn_hidden_size = 4 * hidden_size
    channel = 2256
    expert_num = 16
    dp = 1
    ep = 8
    mp = 16
    net = MoEFFNet(hidden_size, ffn_hidden_size, expert_num, dp, ep, mp, has_bias)
    x = Tensor(np.ones([expert_num, expert_num, channel, hidden_size]), dtype=ms.float16)

    if os.path.exists("./batchmatmul_allreduce_opt/rank_0"):
        shutil.rmtree("./batchmatmul_allreduce_opt/rank_0")

    compile_net(net, x)
    check_output()

    context.set_context(save_graphs=False)
    config = {"enable_allreduce_slice_to_reducescatter": False,}
    with open("./parallel_speed_up_test.json", "w") as file:
        json.dump(config, file, indent=4, separators=(',', ': '))
    context.set_context(
        ascend_config={"parallel_speed_up_json_path": "./parallel_speed_up_test.json"})
