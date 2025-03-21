# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, context
from mindspore.nn import Cell
from mindspore.parallel.shard import Layout
from parallel.utils.utils import compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


class NetSwiGlu(Cell):
    """SwiGlu."""

    def __init__(self, dim=-1, in_strategy=None, out_strategy=None):
        super(NetSwiGlu, self).__init__()
        self.SwigluInfo = ms.ops.auto_generate.gen_ops_prim.Swiglu().shard(in_strategy, out_strategy)
        self.dim = dim

    def construct(self, input_data):
        res = self.SwigluInfo(input_data, self.dim)
        return res


def compile_graph(net, x):
    net.set_train()
    phase = compile_net(net, x)
    return phase


def test_swiglu_auto_parallel():
    """
    Features: test SwigluInfo auto parallel
    Description: auto parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="auto_parallel", search_mode="sharding_propagation", device_num=8,
                                      global_rank=0)
    net = NetSwiGlu()
    input_data = Tensor([[-0.12, 0.123, 31.122, 4.1], [2.1223, 4.1212121217, 0.3123, 8.1]], dtype=ms.float32)
    compile_graph(net, input_data)


def test_swiglu_data_parallel():
    """
    Features: test SwigluInfo data parallel
    Description: data parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    net = NetSwiGlu()
    input_data = Tensor([[-0.12, 0.123, 31.122, 4.1], [2.1223, 4.1212121217, 0.3123, 8.1]], dtype=ms.float32)
    compile_graph(net, input_data)


def test_swiglu_model_parallel():
    """
    Features: test SwigluInfo model parallel
    Description: model parallel
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1),)
    net = NetSwiGlu(dim=0, in_strategy=strategy1)
    input_data = Tensor([[-0.12, 0.123, 31.122, 4.1], [2.1223, 4.1212121217, 0.3123, 8.1]], dtype=ms.float32)
    compile_graph(net, input_data)


def test_swiglu_layout_extend():
    """
    Feature: test SwigluInfo layout extend
    Description: layout extend
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 2, 2), ("dp", "cp", "mp"))
    int_layout = (layout(("dp", "cp"), "None"),)
    net = NetSwiGlu(dim=1, in_strategy=int_layout, out_strategy=None)
    input_data = Tensor(np.zeros((8, 8)), dtype=ms.float32)
    compile_graph(net, input_data)


def test_swiglu_interleaved():
    """
    Feature: test SwigluInfo interleaved
    Description: interleaved
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    layout = Layout((2, 4, 2), ("dp", "cp", "interleaved_parallel"))
    int_layout = (layout(("dp", "interleaved_parallel"), "None"),)
    net = NetSwiGlu(dim=1, in_strategy=int_layout, out_strategy=None)
    input_data = Tensor(np.zeros((8, 8)), dtype=ms.float32)
    compile_graph(net, input_data)


def test_swiglu_extend_dynamic():
    """
    Feature: test dynamic shape for swiglu extend
    Description: no redistribution
    Expectation: compile success
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    strategy1 = ((1, 1),)
    net = NetSwiGlu(dim=0, in_strategy=strategy1)
    input_data = Tensor(shape=[None, 8], dtype=ms.float32)
    net.set_inputs(input_data)
    compile_graph(net, input_data)
