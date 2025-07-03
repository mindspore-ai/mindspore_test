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
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.train import Model
from mindspore.parallel.nn import Pipeline
from mindspore import lazy_inline
from mindspore.nn.utils import no_init_parameters
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore.parallel import build_searched_strategy
from hccl_test.manage.api import Hccl
from .test_pipeline_split import DatasetLenet, MatMulCell


class Net(nn.Cell):
    def __init__(self, strategy1, strategy2, param=None, dtype=ms.float32):
        super().__init__()
        self.block = nn.CellList()
        for i in range(2):
            cell = MatMulCell(strategy1, strategy2, param, dtype)
            cell.pipeline_stage = i
            if i == 1:
                cell.param.requires_grad = False
                cell.param1.requires_grad = False
            self.block.append(cell)

    def construct(self, x):
        for i in range(2):
            x = self.block[i](x)
        return x


class PipelineSplit(nn.Cell):
    @lazy_inline
    def __init__(self, strategy1, strategy2, dtype=ms.float32):
        super().__init__()
        self.cell = Net(strategy1, strategy2, dtype=dtype)

    def construct(self, x, label):
        x = self.cell(x)
        return x


def test_pipeline_split_stage1_save_stra():
    '''
    Feature: pipeline + grad_freeze + stage1 + opt_shard + save_strategy
    Description: In pipeline mode, stage1's param's requires_grad = False, expected success
    Expectation: success
    '''
    hccl = Hccl()
    hccl.rank_id = 16
    hccl.rank_size = 32
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((16, 1), (1, 1))
    strategy2 = ((8, 1), (1, 1))
    with no_init_parameters():
        net = Pipeline(PipelineSplit(strategy1, strategy2), 4, stage_config={"cell.block.0": 0, "cell.block.1": 1})
        params = net.trainable_params()
        optimizer = nn.Lamb(params, learning_rate=0.01)
    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
    parallel_net.hsdp(threshold=1)
    parallel_net.save_param_strategy_file("./strategy_freeze_stage1.ckpt")
    parallel_net.disable_strategy_file_only_for_trainable_params()
    dataset = DatasetLenet(data, label, 3)
    model = Model(parallel_net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    stra = build_searched_strategy("./strategy_freeze_stage1.ckpt")
    assert "cell.block.1.param" in stra
