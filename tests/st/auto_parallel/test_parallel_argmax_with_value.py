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
from mindspore import nn, Tensor, ops
from mindspore.parallel import Layout
from tests.st.auto_parallel.python_shard.utils import global_to_local, local_to_global

D.init()
ms.set_context(pynative_synchronize=True)


def setup_module():
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="Ascend")


class SimpleNet(nn.Cell):
    """Net with Index Select"""

    def construct(self, tensor, axis, keep_dims):
        out, _ = ops.ArgMaxWithValue(axis, keep_dims)(tensor)
        return out, _


def test_parallel_1():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    ms.set_seed(1)
    x = Tensor(np.random.randn(16, 64).astype(np.float32), dtype=ms.float32)

    # Standalone
    net = SimpleNet()
    standalone_output, _ = net(x, axis=0, keep_dims=True)

    # Parallel
    base_device_matrix = (2, 4)
    base_alias_name = ("a", "b")

    layout = Layout(base_device_matrix, base_alias_name)
    x_layout = layout("a", "None")
    x_local = global_to_local(x, x_layout)

    parallel_output, _ = net(x_local, axis=0, keep_dims=True)
    print(parallel_output.shape)
    parallel_output = local_to_global(parallel_output)
    print(parallel_output.shape)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)


def test_parallel_2():
    """
    Feature: Parallel in python shard.
    Description: Test parallel in python shard.
    Expectation: Run success.
    """
    ms.set_seed(1)
    x = Tensor(np.random.randn(16, 64).astype(np.float32), dtype=ms.float32)

    # Standalone
    net = SimpleNet()
    standalone_output, _ = net(x, axis=0, keep_dims=False)

    # Parallel
    base_device_matrix = (2, 4)
    base_alias_name = ("a", "b")

    layout = Layout(base_device_matrix, base_alias_name)
    x_layout = layout("a", "None")
    x_local = global_to_local(x, x_layout)

    parallel_output, _ = net(x_local, axis=0, keep_dims=False)
    print(parallel_output.shape)
    parallel_output = local_to_global(parallel_output)
    print(parallel_output.shape)
    assert np.allclose(standalone_output.asnumpy(), parallel_output.asnumpy(), 1e-3, 1e-3)
