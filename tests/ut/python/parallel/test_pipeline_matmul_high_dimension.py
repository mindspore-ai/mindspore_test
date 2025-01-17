# Copyright 2022 Huawei Technologies Co., Ltd
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
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.nn import PipelineCell, Cell
from mindspore.parallel.shard import Layout

class DatasetLenet():
    def __init__(self, data, length=3):
        self.data = data
        self.index = 1
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data

    @staticmethod
    def get_dataset_size():
        return 32

    @staticmethod
    def get_repeat_count():
        return 1

    @staticmethod
    def get_batch_size():
        return 32

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self

    def reset(self):
        self.index = 0

class TwoDTPCell(Cell):
    def __init__(
            self,
            strategy1=None,
            strategy2=None
    ):
        super().__init__()
        self.mul1 = P.MatMul(transpose_b=False).shard(in_strategy=strategy1)
        self.mul2 = P.MatMul(transpose_b=False).shard(in_strategy=strategy2)
        self.mul1.add_prim_attr("enable_nd_tp", True)
        self.mul2.add_prim_attr("enable_nd_tp", True)
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param")
        self.param2 = Parameter(initializer("zeros", [64, 64]), name="param1")

    def construct(self, x):
        out = self.mul1(x, self.param1)
        out = self.mul2(out, self.param2)
        return out

class Net(nn.Cell):
    def __init__(self, strategy1, strategy2):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = TwoDTPCell(strategy1, strategy2)
            cell.pipeline_stage = i
            self.block.append(cell)

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        return x


def test_pipeline_2dtp():
    """
    Feature:test_pipeline_2dtp
    Description:test_pipeline_2dtp
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=16, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    dataset = DatasetLenet(data, 3)
    layout = Layout((1, 2, 2, 2), ("dp", "cp", "x", "y"))
    strategy1 = (layout("cp", ("x", "y")), layout("x", "y"))
    strategy2 = (layout("cp", ("y", "x")), layout("y", "x"))
    net = PipelineCell(Net(strategy1, strategy2), 4)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)

def test_pipeline_3dtp():
    """
    Feature:test_pipeline_2dtp
    Description:test_pipeline_2dtp
    Expectation:success
    """
    context.set_auto_parallel_context(device_num=16, global_rank=0, pipeline_stages=2)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    dataset = DatasetLenet(data, 3)
    layout = Layout((2, 2, 2, 1), ("cp", "x", "y", "z"))
    strategy1 = (layout(("cp", "z", "y"), "x"), layout(("x", "z"), "y"))
    strategy2 = (layout(("cp", "z", "x"), "y"), layout(("y", "z"), "x"))
    net = PipelineCell(Net(strategy1, strategy2), 4)
    optimizer = nn.Lamb(net.trainable_params(), learning_rate=0.01)
    model = Model(net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
