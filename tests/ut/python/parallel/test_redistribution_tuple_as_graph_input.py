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
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore import context, Tensor
from mindspore.context import ParallelMode
from parallel.utils.utils import compile_net

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def setup_function():
    context.reset_auto_parallel_context()


class ReluNet(nn.Cell):
    def __init__(self, shard):
        super(ReluNet, self).__init__()
        self.relu = P.ReLU().shard((shard,))

    def construct(self, x):
        out_1 = self.relu(x)
        out_2 = out_1 + 1
        return out_1, out_2


class ReshapeNet(nn.Cell):
    def __init__(self):
        super(ReshapeNet, self).__init__()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, x):
        x1, x2 = x
        s1, s2, s3, s4 = self.shape(x1)
        r1 = self.reshape(x1, (s1, s3, s2, s4))
        h1, h2, h3, h4 = self.shape(x2)
        r2 = self.reshape(x2, (h1, h3, h2, h4))
        return r1, r2


class ReshapePartialInputNet(nn.Cell):
    def __init__(self, get_first_input=True):
        super(ReshapePartialInputNet, self).__init__()
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.get_first_input = get_first_input

    def construct(self, x):
        x1, x2 = x
        if self.get_first_input:
            reshape_input = x1
        else:
            reshape_input = x2
        s1, s2, s3, s4 = self.shape(reshape_input)
        r1 = self.reshape(reshape_input, (s1, s3, s2, s4))
        return r1


class TestNet(nn.Cell):
    def __init__(self, from_shard, to_shard):
        super(TestNet, self).__init__()
        self.relu1 = ReluNet((from_shard))
        self.reshape = ReshapeNet()
        self.relu2 = ReluNet((to_shard))
        self.relu1.recompute()
        self.reshape.recompute()
        self.relu2.recompute()
        self.relu1.add_flags(defer_inline=True)
        self.reshape.add_flags(defer_inline=True)
        self.relu2.add_flags(defer_inline=True)

    def construct(self, x):
        out_1 = self.relu1(x)
        s1, s2 = self.reshape(out_1)
        out_2 = self.relu2(s1)
        return out_2, s2


class TestPartialInputNet(nn.Cell):
    def __init__(self, from_shard, to_shard, get_first_input):
        super(TestPartialInputNet, self).__init__()
        self.relu1 = ReluNet((from_shard))
        self.reshape = ReshapePartialInputNet(get_first_input)
        self.relu2 = ReluNet((to_shard))
        self.relu1.recompute()
        self.reshape.recompute()
        self.relu2.recompute()
        self.relu1.add_flags(defer_inline=True)
        self.reshape.add_flags(defer_inline=True)
        self.relu2.add_flags(defer_inline=True)

    def construct(self, x):
        out_1 = self.relu1(x)
        s1 = self.reshape(out_1)
        out_2 = self.relu2(s1)
        return out_2


class TestMultiCallNet(nn.Cell):
    def __init__(self, from_shard, to_shard):
        super(TestMultiCallNet, self).__init__()
        self.relu1 = ReluNet((from_shard))
        self.reshape = ReshapeNet()
        self.relu2 = ReluNet((to_shard))
        self.relu1.add_flags(defer_inline=True)
        self.reshape.add_flags(defer_inline=True)
        self.relu2.add_flags(defer_inline=True)

    def construct(self, x):
        out_1 = self.relu1(x)
        out_2 = self.relu2(x)
        s12 = self.reshape(out_1)
        s34 = self.reshape(out_2)
        s1, s2 = s12
        s3, s4 = s34
        out = s1 + s4 + s2 + s3
        return out


def test_tuple_full_input_redistribution():
    """
    Feature: redistribution that tuple as graph input.
    Description: Test redistribution that tuple as graph input.
    Expectation: Compile success.
    """
    context.reset_auto_parallel_context()
    dataset_shard = (1, 4, 1, 1)
    from_shard = (1, 2, 4, 1)
    to_shard = (1, 4, 1, 2)

    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, dataset_strategy=(dataset_shard,))
    model = TestNet(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[1, 8, 16, 16], dtype=mstype.float16)
    model.set_inputs(input_ids)
    compile_net(model, input_ids)
    context.reset_auto_parallel_context()


def test_tuple_full_input_multi_call_redistribution():
    """
    Feature: redistribution that tuple as graph input. call reshape net twice
    Description: Test redistribution that tuple as graph input. call reshape net twice
    Expectation: Compile success.
    """
    context.reset_auto_parallel_context()
    dataset_shard = (1, 4, 1, 1)
    from_shard = (1, 2, 4, 1)
    to_shard = (1, 4, 1, 2)

    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, dataset_strategy=(dataset_shard,))
    model = TestMultiCallNet(from_shard, to_shard).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[1, 8, 16, 16], dtype=mstype.float16)
    model.set_inputs(input_ids)
    compile_net(model, input_ids)
    context.reset_auto_parallel_context()


def test_tuple_partial_input_redistribution():
    """
    Feature: redistribution that tuple as graph input.
    Description: Test redistribution that tupleas graph input, but only use partial inpupt.
    Expectation: Compile success.
    """
    context.reset_auto_parallel_context()
    dataset_shard = (1, 4, 1, 1)
    from_shard = (1, 2, 4, 1)
    to_shard = (1, 4, 1, 2)

    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, dataset_strategy=(dataset_shard,))
    model = TestPartialInputNet(from_shard, to_shard, True).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[1, 8, 16, 16], dtype=mstype.float16)
    model.set_inputs(input_ids)
    compile_net(model, input_ids)
    context.reset_auto_parallel_context()


def test_tuple_second_input_redistribution():
    """
    Feature: redistribution that tuple as graph input.
    Description: Test redistribution that tupleas graph input, but only use second inpupt.
    Expectation: Compile success.
    """
    context.reset_auto_parallel_context()
    dataset_shard = (1, 4, 1, 1)
    from_shard = (1, 2, 4, 1)
    to_shard = (1, 4, 1, 2)

    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, dataset_strategy=(dataset_shard,))
    model = TestPartialInputNet(from_shard, to_shard, False).to_float(mstype.float16)
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_ids = Tensor(shape=[1, 8, 16, 16], dtype=mstype.float16)
    model.set_inputs(input_ids)
    compile_net(model, input_ids)
    context.reset_auto_parallel_context()
