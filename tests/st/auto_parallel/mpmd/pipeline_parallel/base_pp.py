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
"""simple mlp"""
import time
import numpy as np
import mindspore as ms
from mindspore._c_expression import NoFallbackGuard
from mindspore import nn, Tensor, mint
from mindspore.nn.utils import no_init_parameters
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.parallel.mpmd.pipeline_parallel import Schedule1F1B, PipelineStage
from mindspore.parallel import Layout, hsdp, init_parameters, DTensor
from mindspore.parallel.spmd.shard import shard
from mindspore.common.initializer import initializer


class MLP(nn.Cell):
    """MLP net."""
    def __init__(self, hidden_size, compute_dtype=np.float32, w_layout=None):
        super().__init__()
        # initializing with "ones" is for the convenience of precision compare
        if not w_layout:
            self.weight = ms.Parameter(
                initializer("ones", [hidden_size, hidden_size], ms.float32),
                name="weight")
        else:
            self.weight = ms.Parameter(
                ms.parallel.DTensor.from_local(initializer("ones", [hidden_size, hidden_size], ms.float32), w_layout),
                name="weight")
        self.relu = mint.nn.ReLU()

    def construct(self, x):
        x = mint.matmul(x, self.weight)
        x = self.relu(x)
        return x


class SimpleMLP(nn.Cell):
    """SimpleMLP net."""
    def __init__(self, num_layers, hidden_size, w_layout=None):
        super().__init__()
        self.mlp_layers = nn.CellDict()
        for mlp_id in range(num_layers):
            self.mlp_layers[str(mlp_id)] = MLP(hidden_size, w_layout=w_layout)

    def construct(self, x):
        for mlp in self.mlp_layers.values():
            x = mlp(x)
        return x


def model_split_manual(model, stage_index, stage_num):
    """pipeline parallel split."""
    for i in range(stage_num):
        if i == stage_index:
            continue
        del model.mlp_layers[str(i)]


def check_loss_and_grads(loss, grads, stage_index):
    """validate loss and grads."""
    if stage_index == 3:
        assert np.all(loss[0].asnumpy() == 131072 * 8)
    if stage_index == 0:
        assert np.all(grads[0].asnumpy() == 32768 * 8)
    else:
        assert np.all(grads[0].asnumpy() == 65536 * 4)

def create_dtensor(tensor, layout):
    """create_dtensor"""
    return tensor.local_to_global(layout)


def test_base_pp():
    """
    Feature: HSDP + SHARD + PP.
    Description: Test simple mlp net.
    Expectation: Run success.
    """
    init("hccl")

    # pp config
    num_stages = 4
    micro_batch_num = 8
    rank_id = get_rank()
    device_num = get_group_size()
    device_num_per_stage = device_num // num_stages
    stage_index = rank_id // device_num_per_stage

    # model config
    num_layers = 4
    hidden_size = 32

    # step 1: define network with no init parameters
    with no_init_parameters():
        model = SimpleMLP(num_layers, hidden_size)

    # step 2: retain the net corresponding to this stage
    model_split_manual(model, stage_index, num_stages)

    # step 3: shard
    dp = 1
    mp = 2
    rank_list = [device_num_per_stage * stage_index + i for i in range(device_num_per_stage)]
    layout = Layout((dp, mp), ("dp", "mp"), rank_list)
    if stage_index == 0:
        in_layout = layout("dp", "mp")
        w_layout = layout("mp", "None")
        out_layout = layout("dp", "None")
    else:
        in_layout = layout("dp", "None")
        w_layout = layout("None", "None")
        out_layout = layout("dp", "None")

    strategy = { "forward": { "input": (in_layout,), "output": (out_layout,)},
                 "parameter": {f"{stage_index}.weight": w_layout}}
    shard(model, strategy)

    # step 4: hsdp
    model = hsdp(model, shard_size=2, threshold=0, optimizer_level="level1", enable_grad_accumulation=True)

    # step 5: init parameters
    model = init_parameters(model)

    # step 6: build pp stage
    pipeline_stage = PipelineStage(model, stage_index, num_stages)

    # step 7: select pp scheduler
    schedule = Schedule1F1B(pipeline_stage, micro_batch_num)

    # input
    x_layout = layout("dp", "mp")
    local_batch_size = 8
    local_hidden_size = 16
    x = DTensor.from_local(Tensor(np.ones((local_batch_size, local_hidden_size)), dtype=ms.float32), x_layout)

    optimizer = nn.Adam(model.trainable_params(), learning_rate=0.01)

    # train config
    epochs = 1
    for epoch in range(epochs):
        start = time.time()
        model.zero_grads()
        if stage_index == 0:
            loss, grads = schedule.run(x)
        else:
            loss, grads = schedule.run()
        with NoFallbackGuard():
            optimizer(grads)
        end = time.time()
        print(f"[parallel] Epoch: {epoch+1}/{epochs}, Loss: {loss}, Time: {end - start}")

    check_loss_and_grads(loss, grads, stage_index)
