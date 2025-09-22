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
from mindspore.communication.management import init
from mindspore.parallel import hsdp
from tests.mark_utils import arg_mark
from tests.st.auto_parallel.spmd.common_net import DenseMutiLayerNet

os.environ["MS_SIMULATION_LEVEL"] = "1"
os.environ["RANK_SIZE"] = "32"
os.environ["RANK_ID"] = "32"
init()

loss_fn = nn.MSELoss()

def get_forward_fn(net):
    def forward_fn(data, label):
        logits = net(data)
        loss = loss_fn(logits, label)
        return loss, logits
    return forward_fn

def train(net, data, label):
    grad_fn = ms.value_and_grad(get_forward_fn(net), None, net.trainable_params(), has_aux=True)
    train_steps = 2
    for _ in range(train_steps):
        _, grads = grad_fn(data, label)
        print(grads)

def construct_net_and_data():
    hidden_size = 64
    batch_size = 4
    net = DenseMutiLayerNet(hidden_size, 8)
    data = Tensor(np.random.randn(batch_size, hidden_size).astype(np.float32))
    label = Tensor(np.random.randn(batch_size, hidden_size).astype(np.float32))
    return net, data, label

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_forward_prefetch():
    """
    Feature: hsdp prefetch
    Description: test hsdp forward prefetch
    Expectation: run success
    """
    net, data, label = construct_net_and_data()
    shard_size = 4
    threshold = 0
    for i, layer in enumerate(net.layers):
        hsdp(layer, shard_size, threshold)

    num_layer_to_prefetch = 2
    for i, layer in enumerate(net.layers):
        if i >= len(net.layers) - num_layer_to_prefetch:
            break
        layers_to_perfetch = [net.layers[j] for j in range(i + 1, i + 1 + num_layer_to_prefetch)]
        layer.set_forward_prefetch_cells(layers_to_perfetch)
    train(net, data, label)

@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="essential")
def test_hsdp_backward_prefetch():
    """
    Feature: hsdp prefetch
    Description: test hsdp backward prefetch
    Expectation: run success
    """
    net, data, label = construct_net_and_data()
    shard_size = 4
    threshold = 0
    for i, layer in enumerate(net.layers):
        hsdp(layer, shard_size, threshold)

    num_layer_to_prefetch = 2
    for i, layer in enumerate(net.layers):
        if i < num_layer_to_prefetch:
            continue
        layers_to_perfetch = [net.layers[j] for j in range(i - num_layer_to_prefetch, i)]
        layer.set_backward_prefetch_cells(layers_to_perfetch)
    train(net, data, label)
