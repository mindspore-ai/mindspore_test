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
"""mlp with shared parameter"""
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, mint
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.parallel.mpmd.pipeline_parallel import Schedule1F1B, PipelineStage, SharedParameterInfo
from mindspore.parallel import Layout, DTensor
from mindspore.parallel.spmd.hsdp import hsdp


class MLP(nn.Cell):
    """MLP net."""
    def __init__(self, in_size, out_size, compute_dtype=np.float32, w_layout=None):
        super().__init__()
        # initializing with "ones" is for the convenience of precision compare
        if not w_layout:
            self.weight = ms.Parameter(
                Tensor(np.ones([in_size, out_size]).astype(compute_dtype)),
                name="weight")
        else:
            self.weight = ms.Parameter(
                ms.parallel.DTensor.from_local(Tensor(np.ones([in_size, out_size]).astype(compute_dtype)), w_layout),
                name="weight")
        self.relu = mint.nn.ReLU()

    def construct(self, x):
        x = mint.matmul(x, self.weight)
        x = self.relu(x)
        return x


class MLPWithSharedParam(nn.Cell):
    """MLPWithSharedParam."""
    def __init__(self):
        super().__init__()
        self.relu = mint.nn.ReLU()

    def construct(self, x, param):
        x = mint.matmul(x, param)
        x = self.relu(x)
        return x


class SimpleMLPSingle(nn.Cell):
    """Simple MLP single net."""
    def __init__(self, num_layers, in_size, out_size, compute_dtype=np.float32):
        super().__init__()
        self.mlp_layers = nn.CellDict()
        self.weight = ms.Parameter(
            Tensor(np.ones([in_size, out_size]).astype(compute_dtype)),
            name="shared_weight")
        for mlp_id in range(num_layers - 2):
            self.mlp_layers[str(mlp_id)] = MLP(in_size, out_size)
        self.mlp_with_shared_param = MLPWithSharedParam()

    def construct(self, x):
        x = self.mlp_with_shared_param(x, self.weight)
        for mlp in self.mlp_layers.values():
            x = mlp(x)
        x = self.mlp_with_shared_param(x, self.weight)
        return x


class SimpleMLP(nn.Cell):
    """Simple MLP net."""
    def __init__(self, num_layers, in_size, out_size, w_layout=None):
        super().__init__()
        self.mlp_layers = nn.CellDict()
        for mlp_id in range(num_layers):
            self.mlp_layers[str(mlp_id)] = MLP(in_size, out_size, w_layout=w_layout)

    def construct(self, x):
        for mlp in self.mlp_layers.values():
            x = mlp(x)
        return x


def run_standalone():
    """run standalone."""
    model = SimpleMLPSingle(4, 16, 16)

    def forward_fn(data):
        logits = model(data)
        return logits

    grad_fn = ms.value_and_grad(forward_fn, None, model.trainable_params(), has_aux=False)

    steps = 1
    ret_loss = None
    ret_grads = None
    data = Tensor(np.ones((16, 16)), dtype=ms.float32)
    for _ in range(steps):
        ret_loss, ret_grads = grad_fn(data)
    return ret_loss, ret_grads


def model_split_manual(model, stage_index, stage_num):
    """pipeline parallel split."""
    for i in range(stage_num):
        if i == stage_index:
            continue
        del model.mlp_layers[str(i)]


def run_parallel():
    """run parallel."""
    init("hccl")
    num_stages = 4
    micro_batch_num = 8
    rank_id = get_rank()
    device_num = get_group_size()
    device_num_per_stage = device_num // num_stages
    stage_index = rank_id // device_num_per_stage
    dp = 2
    mp = 1
    layout = Layout((dp, mp), ("dp", "mp"),
                    rank_list=[device_num_per_stage*stage_index + i for i in range(device_num_per_stage)])
    x_layout = layout("dp", "None")
    local_x = DTensor.from_local(Tensor(np.ones((8, 16)), dtype=ms.float32), x_layout)
    w_layout = layout("None", "None")
    model = SimpleMLP(4, 16, 16, w_layout=w_layout)
    model_split_manual(model, stage_index, num_stages)


    shared_params = None
    if stage_index in [0, 3]:
        shared_params = [SharedParameterInfo(model.mlp_layers[str(stage_index)].weight, [0, 3])]

    model = hsdp(model, 2, 0, "level1", True)

    pipeline_stage = PipelineStage(model, stage_index, num_stages, shared_parameters=shared_params)
    schedule = Schedule1F1B(pipeline_stage, micro_batch_num)

    epochs = 1
    loss = None
    grads = None
    for _ in range(epochs):
        model.zero_grads()
        if stage_index == 0:
            loss, grads = schedule.run(local_x)
        else:
            loss, grads = schedule.run()
    return loss, grads


def test_pp_with_shared_param():
    """
    Feature: HSDP + SHARD + PP + SharedParameter.
    Description: Test simple mlp net.
    Expectation: Run success.
    """
    standalone_loss, standalone_grads = run_standalone()
    pp_loss, pp_grads = run_parallel()
    num_stages = 4
    rank_id = get_rank()
    device_num = get_group_size()
    device_num_per_stage = device_num // num_stages
    stage_index = rank_id // device_num_per_stage
    if stage_index == 3:
        assert np.allclose(standalone_loss[:1, :].asnumpy(), pp_loss[0].asnumpy())
    expect_grad = None
    if stage_index in [1, 2]:
        expect_grad = standalone_grads[stage_index].asnumpy()
    else:
        expect_grad = standalone_grads[0].asnumpy()
    assert np.allclose(expect_grad[:8, :], pp_grads[0].asnumpy())
