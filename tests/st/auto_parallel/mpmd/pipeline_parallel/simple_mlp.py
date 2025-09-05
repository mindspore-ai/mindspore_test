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
import time
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, mint
from mindspore.communication.management import init, get_rank, get_group_size, create_group
from mindspore.parallel.mpmd.pipeline_parallel import Schedule1F1B, PipelineStage, P2PInfo
from mindspore.parallel import Layout
from mindspore.parallel.spmd.hsdp import hsdp


class MLP(nn.Cell):
    def __init__(self, hidden_size, compute_dtype=np.float32):
        super().__init__()
        # initializing with "ones" is for the convenience of precision compare
        self.weight = ms.Parameter(
            Tensor(np.ones([hidden_size, hidden_size]).astype(compute_dtype)),
            name="weight")
        self.relu = mint.nn.ReLU()

    def construct(self, x):
        x = mint.matmul(x, self.weight)
        x = self.relu(x)
        return x


class SimpleMLP(nn.Cell):
    def __init__(self, num_layers, hidden_size):
        super().__init__()
        self.mlp_layers = nn.CellDict()
        for mlp_id in range(num_layers):
            self.mlp_layers[str(mlp_id)] = MLP(hidden_size)

    def construct(self, x):
        for mlp in self.mlp_layers.values():
            x = mlp(x)
        return x


def model_split_manual(model, stage_index, stage_num):
    for i in range(stage_num):
        if i == stage_index:
            continue
        del model.mlp_layers[str(i)]


def build_send_recv_info(stage_index, per_batch_size, hidden_size, layout):
    recv_info_list = []
    send_info_list = []
    if stage_index == 0:
        send_info = P2PInfo(shape=[per_batch_size, hidden_size], dtype=ms.float32)
        send_info_list.append(send_info)
    elif stage_index == 3:
        # for forward recv
        recv_info = P2PInfo(shape=[per_batch_size, hidden_size], dtype=ms.float32, layout=layout)
        recv_info_list.append(recv_info)
    else:
        # for forward recv
        recv_info = P2PInfo(shape=[per_batch_size, hidden_size], dtype=ms.float32, layout=layout)
        recv_info_list.append(recv_info)
        # for forward send
        send_info = P2PInfo(shape=[per_batch_size, hidden_size], dtype=ms.float32, layout=layout)
        send_info_list.append(send_info)
    return send_info_list, recv_info_list


def check_loss_and_grads(loss, grads, stage_index):
    if stage_index == 3:
        assert np.all(loss[0].asnumpy() == 131072)
    if stage_index == 0:
        assert np.all(grads[0].asnumpy() == 32768)
    else:
        assert np.all(grads[0].asnumpy() == 65536)


def test_simple_mlp():
    """
    Feature: HSDP + SHARD + PP.
    Description: Test simple mlp net.
    Expectation: Run success.
    """
    ms.set_seed(0)
    init("hccl")

    # pp config
    num_stages = 4
    micro_batch_num = 8
    rank_id = get_rank()
    device_num = get_group_size()
    device_num_per_stage = device_num // num_stages
    stage_index = rank_id // device_num_per_stage
    rank_ids = [rank_id + device_num_per_stage * (i - stage_index) for i in range(num_stages)]
    # if the names are the same, an error will be reported
    pp_group = f"pipeline_group_{rank_id%2}"
    create_group(pp_group, rank_ids)

    # model config
    local_batch_size = 8
    num_layers = 4
    local_hidden_size = 16
    model = SimpleMLP(num_layers, local_hidden_size)
    model_split_manual(model, stage_index, num_stages)

    # shard config
    dp = 1
    mp = 2
    layout = Layout((dp, mp), ("dp", "mp"),
                    rank_list=[device_num_per_stage*stage_index + i for i in range(device_num_per_stage)])
    x_layout = layout("dp", "mp")
    local_x = Tensor(np.ones((local_batch_size, local_hidden_size)), dtype=ms.float32)
    local_x.local_to_global(x_layout)
    if stage_index == 0:
        w_layout = layout("mp", "None")
        # handle partial
        relu_layout = layout("dp", "None")
        model.mlp_layers[str(stage_index)].relu.shard(in_strategy=(relu_layout,))
    else:
        w_layout = layout("None", "None")
    model.mlp_layers[str(stage_index)].weight.local_to_global(w_layout)

    #hsdp
    shard_size = 2
    threshold = 0
    learning_rate = 0.01
    optimizer_level = "level1"
    enable_accu = True
    model = hsdp(model, shard_size, threshold, optimizer_level, enable_accu)
    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)

    # pp stage
    recv_layout = layout("dp", "None")
    send_info_list, recv_info_list = build_send_recv_info(stage_index, local_batch_size//micro_batch_num,
                                                          local_hidden_size, layout=recv_layout)
    pipeline_stage = PipelineStage(model, stage_index, num_stages, pp_group,
                                   recv_info=recv_info_list, send_info=send_info_list)

    schedule = Schedule1F1B(pipeline_stage, micro_batch_num)

    # train config
    epochs = 1
    for epoch in range(epochs):
        start = time.time()
        model.zero_grads()
        if stage_index == 0:
            loss, grads = schedule.run(local_x)
        else:
            loss, grads = schedule.run()
        optimizer(grads)
        end = time.time()
        print(f"[parallel] Epoch: {epoch+1}/{epochs}, Loss: {loss}, Time: {end - start}")

    check_loss_and_grads(loss, grads, stage_index)
