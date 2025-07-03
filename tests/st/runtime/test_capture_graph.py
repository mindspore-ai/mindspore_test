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

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.runtime as rt
from mindspore import Tensor, context, mutable
from mindspore.common import Parameter
from mindspore import dtype as mstype
from tests.mark_utils import arg_mark

context.set_context(
    mode=context.GRAPH_MODE,
    jit_config={
        "jit_level": "O0",
        "infer_boost": "on"
    },
    max_call_depth=600000
)

g_block_num = 20
steps = 20
input_len = 10

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(Tensor(2, ms.float32))
        self.add = P.Add()
        self.mul = P.Mul()

    def construct(self, x):
        x = self.add(x, self.param)
        for _ in range(5):
            x = self.add(x, 0.1)
            x = self.add(x, 0.2)
        x = self.mul(x, 2)
        x = self.add(x, 0.5)
        return x

class SeqNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = Net()

    def construct(self, x):
        output = self.net(x)
        return output

class Net1(nn.Cell):
    def __init__(self):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.add_n = P.AddN()
        self.reshape = P.Reshape()

    def construct(self, x, key_cache_list, value_cache_list):
        y = x
        x = self.reshape(x, (1, -1))
        for i in range(g_block_num):
            key = key_cache_list[int(i/2) % input_len]
            x = self.add(x, 1)
            x = self.sub(x, 1.1)
            x = self.reshape(x, (2, -1))
            x = self.add(x, key)
            x = self.add(x, y)
            x = self.mul(x, 0.251)
            x = self.add(x, 1)
            x = self.add(x, key)
            x = self.mul(x, 0.501)
            x = self.sub(x, 1.1)
            x = self.reshape(x, (2, -1))
            x = self.mul(x, 2)
            x = self.add(x, y)
            x = self.sub(x, 1.1)
            x = self.sub(x, key)
            x = self.reshape(x, (4, -1))
            x = self.mul(x, 0.051)
            x = self.reshape(x, (2, -1))
            x = self.add_n(value_cache_list) + y + x
            x = self.add(x, key)
        x = self.reshape(x, (2, -1))
        return x

def expected_output(x):
    return (x + 3.5) * 2 + 0.5

@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
def test_dynamic_shape_for_capture_graph():
    """
    Feature: graph mode support capture graph
    Description: Test dynamic shape scene and dyn value for capture graph
    Expectation: No exception and result is correct
    """
    rt.set_kernel_launch_capture(True)
    new_input1 = Tensor(np.ones((2, 5)).astype(np.float32))
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
    base_shape = (2, 3)

    net = SeqNet()
    net.set_inputs(dyn_input_data)
    net.phase = "increment"

    for i in range(1, 20):
        if i == 5:
            output = net(new_input1)
            output_np = output.asnumpy()
        else:
            input_data1 = Tensor(np.full(base_shape, i).astype(np.float32))
            output = net(input_data1)
            output_np = output.asnumpy()
            expected = expected_output(i)
            assert np.allclose(output_np, expected), \
                f"Output {output_np} does not match expected {expected} at step {i}"

@arg_mark(
    plat_marks=['platform_ascend910b'],
    level_mark='level0',
    card_mark='onecard',
    essential_mark='essential'
)
def test_kv_cache_for_capture_graph():
    """
    Feature: graph mode support capture graph
    Description: Test kv_cache scene
    Expectation: No exception and result is correct
    """
    rt.set_kernel_launch_capture(True)
    input_data1 = Tensor(np.zeros((2, 2)).astype(np.float32))
    input_data2 = Tensor(np.zeros((2, 4)).astype(np.float32))
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
    k_cache_list1 = []
    v_cache_list1 = []
    k_cache_list2 = []
    v_cache_list2 = []
    dyn_k_cache_list = []
    dyn_v_cache_list = []

    for _ in range(input_len):
        dyn_k_cache_list.append(dyn_input_data)
        dyn_v_cache_list.append(dyn_input_data)

    for _ in range(input_len):
        new_input_data = P.Add()(input_data1, 1)
        k_cache_list1.append(new_input_data)
        v_cache_list1.append(new_input_data)

    net = Net1()
    net.set_inputs(dyn_input_data, mutable(dyn_k_cache_list), mutable(dyn_v_cache_list))
    net.phase = "increment"

    output = net(input_data1, mutable(k_cache_list1), mutable(v_cache_list1))
    output = net(input_data1, mutable(k_cache_list1), mutable(v_cache_list1))

    k_cache_list1 = []
    v_cache_list1 = []

    for _ in range(input_len):
        new_input_data = P.Add()(input_data2, 1)
        k_cache_list2.append(new_input_data)
        v_cache_list2.append(new_input_data)

    for _ in range(steps):
        output = net(input_data2, mutable(k_cache_list2), mutable(v_cache_list2))
        output.asnumpy()

    expected = np.array([[11.036, 11.036, 11.036, 11.036], [11.036, 11.036, 11.036, 11.036]], dtype=np.float32)

    assert np.allclose(output, expected, rtol=0, atol=0.001), f"Result wrong, real: {output}, expected: {expected}"
