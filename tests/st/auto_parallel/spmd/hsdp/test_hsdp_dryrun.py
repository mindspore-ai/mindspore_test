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
import os
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor
from mindspore.parallel import Layout
from mindspore.communication.management import init
from mindspore.parallel.spmd.hsdp.hsdp import hsdp
os.environ["MS_SIMULATION_LEVEL"] = "0"
os.environ["RANK_SIZE"] = "32"
os.environ["RANK_ID"] = "0"
init()

loss_fn = nn.MSELoss()

def get_forward_fn(net):
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    return forward_fn

def run_hsdp_with_layout(w_layout, data_layout, label_layout):
    """
    hsdp with layout
    """
    in_channels = 256
    out_channels = 64
    net = nn.Dense(in_channels, out_channels, weight_init="ones")

    batch_size = 4
    local_data = Tensor(np.random.randn(batch_size, in_channels).astype(np.float32))
    local_label = Tensor(np.random.randn(batch_size, out_channels).astype(np.float32))

    global_data = local_data.local_to_global(data_layout)
    global_label = local_label.local_to_global(label_layout)
    net.weight = net.weight.local_to_global(w_layout)

    shard_size = 4
    threshold = 0
    optimizer_level = "level1"
    hsdp(net, shard_size, threshold, optimizer_level)

    optimizer = nn.Adam(net.trainable_params(), 1e-2)
    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)

    train_steps = 2
    for _ in range(train_steps):
        _, grads = grad_fn(global_data, global_label)
        optimizer(grads)

def get_device_layout():
    device_matrix = (4, 8)
    alias_name = ("dp", "mp")
    rank_list = list(range(32))
    layout = Layout(device_matrix, alias_name, rank_list)
    return layout

def hsdp_with_dp_layout():
    """
    Feature: hsdp
    Description: test hsdp with data parallel layout
    Expectation: compile success
    """
    layout = get_device_layout()
    w_layout = layout("None", "None")
    data_layout = layout("dp", "None")
    label_layout = layout("dp", "None")
    run_hsdp_with_layout(w_layout, data_layout, label_layout)

def hsdp_with_mp_layout():
    """
    Feature: hsdp
    Description: test hsdp with model parallel layout
    Expectation: compile success
    """
    layout = get_device_layout()
    w_layout = layout("mp", "None")
    data_layout = layout("dp", "None")
    label_layout = layout("dp", "None")
    run_hsdp_with_layout(w_layout, data_layout, label_layout)
