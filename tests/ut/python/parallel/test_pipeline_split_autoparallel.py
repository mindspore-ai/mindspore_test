# Copyright 2020 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model
from mindspore.parallel.nn import Pipeline, MicroBatchInterleaved
from mindspore.nn.wrap.cell_wrapper import Cell, _MicroBatch
from mindspore.nn.utils import no_init_parameters
from mindspore import lazy_inline
from mindspore import ParameterTuple
from mindspore.parallel.auto_parallel import AutoParallel
from hccl_test.manage.api import Hccl

class SimpleNet(nn.Cell):
    def __init__(self, matmul_weight):
        super().__init__()
        self.matmul1 = P.MatMul()
        self.matmul2 = P.MatMul()
        self.matmul1_w = Parameter(matmul_weight[0], name="weight1")
        self.matmul2_w = Parameter(matmul_weight[1], name="weight2")

    def construct(self, inputs):
        x = self.matmul1(inputs, self.matmul1_w)
        x = self.matmul2(x, self.matmul2_w)
        return x


class StageSimpleNet(nn.Cell):
    def __init__(self, w_l, micro, stage_num=2):
        super().__init__()
        self.micro_size = micro
        self.block = nn.CellList()
        self.add = P.TensorAdd()
        self.w_l = w_l
        self.add_list = []
        self.relu_block = nn.CellList()
        for i in range(self.micro_size):
            cell = SimpleNet(w_l[i])
            relu = nn.ReLU()
            if self.micro_size > stage_num:
                cell.pipeline_stage = i // 2
                relu.pipline_stage = i // 2
            else:
                cell.pipeline_stage = i
                relu.pipline_stage = i
            self.relu_block.append(relu)
            self.block.append(cell)
            self.add_list.append(Parameter(Tensor(np.full((1, 16), 0.1, dtype=np.float32)), name=f"weight{i}"))
        self.add_tuple = ParameterTuple(self.add_list)

    def construct(self, x):
        for i in range(self.micro_size):
            x = self.block[i](x)
            x = self.relu_block[i](x)
            x = self.add(x, self.add_tuple[i])
        return x


class StageSimpleWithLazyInlineNet(nn.Cell):
    @lazy_inline
    def __init__(self, w_l, micro, stage_num=2):
        super().__init__()
        self.micro_size = micro
        self.block = nn.CellList()
        self.add = P.TensorAdd()
        self.w_l = w_l
        self.add_list = []
        self.relu_block = nn.CellList()
        for i in range(self.micro_size):
            cell = SimpleNet(w_l[i])
            relu = nn.ReLU()
            if self.micro_size > stage_num:
                cell.pipeline_stage = i // 2
                relu.pipline_stage = i // 2
            else:
                cell.pipeline_stage = i
                relu.pipline_stage = i
            self.relu_block.append(relu)
            self.block.append(cell)
            self.add_list.append(
                Parameter(Tensor(np.full((1, 16), 0.1, dtype=np.float32)), name=f"weight{i}"))
        self.add_tuple = ParameterTuple(self.add_list)

    def construct(self, x):
        for i in range(self.micro_size):
            x = self.block[i](x)
            x = self.relu_block[i](x)
            x = self.add(x, self.add_tuple[i])
        return x


class DatasetLenet():
    def __init__(self, data, label, length=3):
        self.data = data
        self.label = label
        self.index = 1
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data, self.label

    def reset(self):
        self.index = 0

    def get_dataset_size(self):
        return 32

    def get_repeat_count(self):
        return 1

    def get_batch_size(self):
        return 32

    def create_tuple_iterator(self, num_epochs=1, do_copy=True):
        return self


class MatMulCell(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None, dtype=ms.float32):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        if param is not None:
            self.param = param
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul().shard(strategy1)
        self.matmul1 = P.MatMul().shard(strategy2)
        self.cast = P.Cast()
        self.dtype = dtype

    def construct(self, x):
        out = self.matmul(self.cast(x, self.dtype), self.cast(self.param, self.dtype))
        out = self.matmul1(out, self.cast(self.param1, self.dtype))
        return out


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None, dtype=ms.float32):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = MatMulCell(strategy1, strategy2, param, dtype)
            cell.pipeline_stage = i
            self.block.append(cell)

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        return x


class PipelineSplit(nn.Cell):
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.cell = Net(strategy1, strategy2, dtype=dtype)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)

    def construct(self, x, label):
        x = self.cell(x)
        return x

class PipelineSplitLazyInline(nn.Cell):
    @lazy_inline
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.cell = Net(strategy1, strategy2, dtype=dtype)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)

    def construct(self, x, label):
        x = self.cell(x)
        return x

class PipelineSplit2(nn.Cell):
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.cell = Net(strategy1, strategy2, self.param, dtype)
        self.cell.block[0].matmul.add_prim_attr("parameter_start", 0)

    def construct(self, x, label):
        x = self.cell(x)
        return x


class PipelineDupCell(Cell):
    def __init__(self, network, micro_size):
        super(PipelineDupCell, self).__init__(auto_prefix=False)
        self.network = network
        self.micro_inputs = nn.CellList()
        self.micro_size = micro_size
        self.add_list = []
        for _ in range(micro_size):
            micro_input = _MicroBatch(micro_size)
            self.micro_inputs.append(micro_input)
            self.add = P.Add()
            self.add_list.append(self.add)

    def construct(self, *inputs):
        ret = None
        for i in range(self.micro_size):
            micro_input = self.micro_inputs[i](i, *inputs)
            output = self.network(*micro_input)
            if ret is not None:
                ret = self.add_list[i](ret, output)
            else:
                ret = output
        return ret

def test_pipeline_split_with_micro_batch_interleaved_stage0_1():
    """
    Feature: test PipelineSplit with MicroBatchInterleaved in auto parallel.
    Description: net with MicroBatchInterleaved in semi auto parallel.
    Expectation: success.
    """
    hccl = Hccl()
    hccl.rank_id = 0
    hccl.rank_size = 32
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    micro_batch_interleaved = 2
    with no_init_parameters():
        net = Pipeline(MicroBatchInterleaved(PipelineSplit(strategy1, strategy2), micro_batch_interleaved), 4,
                       stage_config={"network.cell.block.0": 0, "network.cell.block.1": 1})
        params = net.network.network.cell.block[0].trainable_params()
        optimizer = nn.Lamb(params, learning_rate=0.01)
    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
    parallel_net.pipeline(stages=2)
    dataset = DatasetLenet(data, label, 3)
    model = Model(parallel_net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
