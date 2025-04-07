# Copyright 2023-2025 Huawei Technologies Co., Ltd
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
import os

import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.nn import PipelineCell, Cell
from mindspore import lazy_inline
from mindspore.communication import init, get_rank
from mindspore.nn.optim import Momentum
import mindspore.dataset as ds
from mindspore import log as logger


context.set_context(mode=context.GRAPH_MODE)
init()
ms.set_seed(1)


class MyIter:
    def __init__(self, data_input, length=1):
        self.data = data_input
        self.index = 1
        self.length = length

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.data

    def __iter__(self):
        self.index = 0
        return self

    def __len__(self):
        return self.length

    def reset(self):
        self.index = 0


class MatMulCell(Cell):
    def __init__(self):
        super().__init__()
        self.param = Parameter(initializer("zeros", [64, 64]), name="param")
        self.param1 = Parameter(initializer("zeros", [64, 64]), name="param1")
        self.matmul = P.MatMul()
        self.matmul1 = P.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.param)
        out = self.matmul1(out, self.param1)
        return out


class Net(nn.Cell):
    @lazy_inline
    def __init__(self):
        super().__init__()
        self.block = nn.CellList()
        for i in range(8):
            cell = MatMulCell()
            cell.pipeline_stage = i
            self.block.append(cell)
        self.block[3].recompute()

    def construct(self, x):
        for i in range(8):
            x = self.block[i](x)
        return x


# save checkpoint when model train
def model_train(model, epoch, dataset, ckpt_path, ckpt_prefix, integral_save, remove_redundancy):
    ckpt_config = CheckpointConfig(save_checkpoint_steps=1,
                                   keep_checkpoint_max=5,
                                   integrated_save=integral_save,
                                   async_save=False,
                                   remove_redundancy=remove_redundancy)
    loss_cb = LossMonitor(1)
    ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)
    model.train(epoch=epoch, train_dataset=dataset, callbacks=[ckpt_callback, loss_cb], dataset_sink_mode=False)


def test_compile_cache_in_parallel_mode():
    """
    Feature: enable MS_COMPILER_CACHE_ENABLE.
    Description: the is_param_init property of ParamInfo is true.
    Expectation: compile pass.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", pipeline_stages=8)

    data1 = Tensor(np.ones([32, 64]), dtype=ms.float32)
    net = PipelineCell(Net(), 8)
    learning_rate = 0.01
    momentum = 0.9
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    parallel_model = Model(net, optimizer=optimizer)

    from mindspore.train.serialization import _is_auto_parallel_mode
    flag1 = _is_auto_parallel_mode(parallel_model.train_network)
    logger.warning("Before train, the param is inited in parallel_mode, it is %s." % flag1)

    parallel_dataset = ds.GeneratorDataset(source=MyIter(data1, 1), column_names=["data"])
    parallel_model.build(parallel_dataset)

    # When caching is enabled, parm init should be True in parallel mode.
    exec_path = os.path.dirname(os.path.realpath(__file__))
    temp_dir = os.path.join(exec_path, "test_run_compile_cache_mp")
    cpkt_root_dir = os.path.join(temp_dir, "cpkt_dir")
    cur_ckpt_path = f"{cpkt_root_dir}/rank_{get_rank()}_ckpt"
    model_train(model=parallel_model, epoch=1, dataset=parallel_dataset, ckpt_path=cur_ckpt_path, \
                ckpt_prefix="parallel", integral_save=False, remove_redundancy=False)

    flag2 = _is_auto_parallel_mode(parallel_model.train_network)
    logger.warning("After train, the param is inited in parallel_mode, it is %s." % flag2)
