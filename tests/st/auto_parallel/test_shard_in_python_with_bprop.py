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

import numpy as np
import mindspore as ms
import mindspore.communication.management as D
from mindspore import nn, Tensor
from mindspore.parallel import Layout

ms.set_context(mode=ms.PYNATIVE_MODE)
D.init()
class SimpleModel(nn.Cell):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = ms.Parameter(
            Tensor(np.random.randn(input_size, output_size).astype(np.float32)),
            name='weight'
        )

    def construct(self, x):
        x = ms.mint.matmul(x, self.weight)
        x = ms.mint.nn.ReLU()(x)
        x = x + 1
        return x

def create_dtensor(data, layout):
    """create_dtensor"""
    tensor = Tensor(data, dtype=ms.float32)
    return tensor.local_to_global(layout)

def print_layout_info(tensor, name):
    """print_layout_info"""
    if hasattr(tensor, 'layout') and tensor.layout is not None:
        layout_dict = tensor.layout.to_dict()
        print(f"{name} Layout:")
        print(f"  device_matrix: {layout_dict['device_matrix']}")
        print(f"  tensor_map: {layout_dict['tensor_map']}")
        print(f"  alias_name: {layout_dict['alias_name']}")
        print(f"  rank_list: {layout_dict['rank_list'][:8]}...")  # 只显示前8个rank
    else:
        print(f"{name} has no layout information")

def run_network(x_layout, w_layout, target_layout):
    input_size = 10
    output_size = 5
    batch_size = 4
    learning_rate = 0.01
    epochs = 10

    model = SimpleModel(input_size, output_size)
    loss_fn = nn.MSELoss()

    def forward_fn(data, label):
        logits = model(data)
        loss = loss_fn(logits, label)
        return loss, logits

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
    np_x = np.random.randn(batch_size, input_size).astype(np.float32)
    np_target = np.random.randn(batch_size, output_size).astype(np.float32)
    x = create_dtensor(np_x, x_layout)
    target = create_dtensor(np_target, target_layout)
    print_layout_info(x, "Input X")
    model.weight = model.weight.local_to_global(w_layout)
    print_layout_info(model.weight, "Input w")
    print_layout_info(target, "Input target")
    for epoch in range(epochs):
        (loss_value, grads) = grad_fn(x, target)
        optimizer(grads)
        print(f"Epoch: {epoch+1}/{epochs}, Loss: {loss_value}")


base_device_matrix = (2, 4)  # dp=2, mp=4
base_alias_name = ("dp", "mp")
base_rank_list = list(range(8))

def test_data_parallel():
    '''
    Feature: Data parallel in python shard.
    Description: Test data parallel in python shard.
    Expectation: Run success.
    '''
    x_layout1 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout1 = x_layout1("dp", "None")

    w_layout1 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout1 = w_layout1("None", "None")

    target_layout1 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    target_layout1 = target_layout1("dp", "None")
    run_network(x_layout1, w_layout1, target_layout1)

def test_tensor_parallel():
    '''
    Feature: Tensor parallel in python shard.
    Description: Test tensor parallel in python shard.
    Expectation: Run success.
    '''
    x_layout1 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    x_layout1 = x_layout1("dp", "mp")

    w_layout1 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    w_layout1 = w_layout1("mp", "None")

    target_layout1 = Layout(base_device_matrix, base_alias_name, base_rank_list)
    target_layout1 = target_layout1("dp", "None")
    run_network(x_layout1, w_layout1, target_layout1)
