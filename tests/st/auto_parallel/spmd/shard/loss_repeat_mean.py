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
import mindspore.communication.management as D
from mindspore import nn, Tensor
from mindspore.parallel import Layout


class SimpleModel(nn.Cell):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.weight = ms.Parameter(
            Tensor(np.ones([input_size, output_size]).astype(np.float32)),
            name='weight'
        )

        self.relu = ms.mint.nn.ReLU()

    def construct(self, x):
        x = ms.mint.matmul(x, self.weight)
        x = self.relu(x)
        return x


def create_dtensor(data, layout):
    """create_dtensor"""
    tensor = Tensor(data, dtype=ms.float32)
    return tensor.local_to_global(layout)


def create_tensor(data):
    return Tensor(data, dtype=ms.float32)


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


def run_standalone(x, input_size, output_size, learning_rate=0.01, epochs=2):
    model = SimpleModel(input_size, output_size)

    def forward_fn(data):
        logits = model(data)
        return logits

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    x = create_tensor(x)

    ret_loss = None
    ret_grads = None
    for epoch in range(epochs):
        start = time.time()
        (loss_value, grads) = grad_fn(x)
        optimizer(grads)
        end = time.time()
        ret_loss = loss_value
        ret_grads = grads
        print(f"[standalone] Epoch: {epoch+1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def run_parallel(local_x, local_input_size, local_output_size, x_layout, w_layout, relu_strategy, learning_rate=0.01,
                 epochs=2):
    model = SimpleModel(local_input_size, local_output_size)

    def forward_fn(data):
        logits = model(data)
        return logits

    optimizer = nn.Adam(model.trainable_params(), learning_rate=learning_rate)
    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=False)

    x = create_dtensor(local_x, x_layout)
    model.weight = model.weight.local_to_global(w_layout)
    model.relu.shard(in_strategy=relu_strategy[0], out_strategy=relu_strategy[1])

    ret_loss = None
    ret_grads = None
    for epoch in range(epochs):
        start = time.time()
        (loss_value, grads) = grad_fn(x)
        optimizer(grads)
        end = time.time()
        ret_loss = loss_value
        ret_grads = grads
        print(f"[parallel] Epoch: {epoch+1}/{epochs}, Loss: {loss_value}, Time: {end - start}")

    return ret_loss, ret_grads


def test_loss_repeat_mean():
    '''
    Feature: loss repeat mean.
    Description: Test loss repeat mean.
    Expectation: Run success.
    '''
    D.init()

    # standalone
    batch_size = 4
    input_size = 32
    output_size = 2
    batch_size = 4

    x = np.ones([batch_size, input_size]).astype(np.float32)
    standalone_loss, _ = run_standalone(x, input_size, output_size)

    # parallel
    dp = 1
    mp = 8
    local_batch_size = batch_size // dp
    local_input_size = input_size // mp
    local_output_size = output_size
    local_x = np.ones([local_batch_size, local_input_size]).astype(np.float32)
    layout = Layout((dp, mp), ("dp", "mp"))
    x_layout = layout("dp", "mp")
    w_layout = layout("mp", "None")
    relu_strategy = ((layout("None", "None"),), (layout("None", "None"),))
    parallel_loss, _ = run_parallel(local_x, local_input_size, local_output_size, x_layout, w_layout, relu_strategy)

    # compare
    assert np.allclose(standalone_loss.asnumpy(), parallel_loss.asnumpy(), 0.001, 0.001)
