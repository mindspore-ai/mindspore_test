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
import mindspore.nn as nn
import mindspore.ops as P
from mindspore import Tensor, mutable
from mindspore import dtype as mstype
import mindspore.context as context
from tests.mark_utils import arg_mark


g_block_num = 50
steps = 50

ascend_home_path = os.getenv('ASCEND_HOME_PATH')
if not ascend_home_path:
    os.environ['ASCEND_HOME_PATH'] = "/usr/local/Ascend/latest"

ms.runtime.set_kernel_launch_group()
context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0", "infer_boost": "on"}, max_call_depth=6000)

class Net(nn.Cell):
    """
    Construct a single-input network structure.
    """
    def __init__(self):
        super().__init__()
        self.add = P.Add()
        self.mul = P.Mul()
        self.sub = P.Sub()
        self.add_n = P.AddN()
        self.reshape = P.Reshape()

    def construct(self, x, key_cache_list, value_cache_list):
        x = self.reshape(x, (1, -1))

        for _ in range(g_block_num):
            x = self.add(x, 1)
            x = self.sub(x, 1.1)
            x = self.reshape(x, (2, -1))
            x = self.mul(x, 0.251)
            x = self.add(x, 1)

            x = self.mul(x, 0.501)
            x = self.sub(x, 1.1)
            x = self.reshape(x, (2, -1))
            x = self.mul(x, 2)
            x = self.add(x, 1)
            x = self.sub(x, 1.1)
            x = self.reshape(x, (4, -1))
            x = self.mul(x, 0.051)
            x = self.reshape(x, (2, -1))
            x = self.add_n(key_cache_list) + x
            x = self.add_n(value_cache_list) + x

        x = self.reshape(x, (2, -1))
        return x


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_host_bound_for_parallel_dispatch():
    """
    Feature: Graph parallel dispatch kernel.
    Description: Test host bound case for parallel dispatch kernels, which includes aclnn and inference
                 internal kernels.
    Expectation: The program execute and exit normally.
    """
    input_data1 = Tensor(np.zeros((2, 2)).astype(np.float32))
    input_data2 = Tensor(np.zeros((2, 4)).astype(np.float32))
    dyn_input_data = Tensor(shape=[2, None], dtype=mstype.float32)
    k_cache_list1 = []
    v_cache_list1 = []
    k_cache_list2 = []
    v_cache_list2 = []
    dyn_k_cache_list = []
    dyn_v_cache_list = []

    for _ in range(10):
        dyn_k_cache_list.append(dyn_input_data)
        dyn_v_cache_list.append(dyn_input_data)

    for _ in range(10):
        new_input_data = P.Add()(input_data1, 1)
        k_cache_list1.append(new_input_data)
        v_cache_list1.append(new_input_data)

    net = Net()
    net.set_inputs(dyn_input_data, mutable(dyn_k_cache_list), mutable(dyn_v_cache_list))
    net.phase = "increment"


    # warm up
    output = net(input_data1, mutable(k_cache_list1), mutable(v_cache_list1))
    output = net(input_data1, mutable(k_cache_list1), mutable(v_cache_list1))
    print(output)
    k_cache_list1 = []
    v_cache_list1 = []

    for _ in range(10):
        new_input_data = P.Add()(input_data2, 1)
        k_cache_list2.append(new_input_data)
        v_cache_list2.append(new_input_data)

    for _ in range(steps):
        output = net(input_data2, mutable(k_cache_list2), mutable(v_cache_list2))
        output.asnumpy()

    exp_val = 20.191507
    exp_array = np.array([[exp_val, exp_val, exp_val, exp_val], [exp_val, exp_val, exp_val, exp_val]])
    assert np.allclose(output.asnumpy(), exp_array, 0.0001, 0.0001)
