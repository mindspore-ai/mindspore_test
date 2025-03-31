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
""" test group info """
import os
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.nn.utils import no_init_parameters
from mindspore.parallel.auto_parallel import AutoParallel
from mindspore import context
from mindspore import restore_group_info_list
from mindspore.train import Model
from mindspore.parallel.nn import Pipeline
from hccl_test.manage.api import Hccl
from .test_pipeline_split import PipelineSplit, DatasetLenet

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")


def test_pipeline_split_stage1_mirror_group():
    """
    Feature: save and load mirror group
    Description: semi-auto parallel, pipeline parallel.
    Expectation: group info list match expectation value.
    """
    hccl = Hccl()
    hccl.rank_id = 63
    hccl.rank_size = 64
    os.environ['GROUP_INFO_FILE'] = "./test_pipeline_split_stage1_mirror_group.pb"
    data = Tensor(np.ones([32, 64]), dtype=ms.float32)
    label = Tensor(np.ones([64, 64]), dtype=ms.float32)
    strategy1 = ((4, 1), (1, 8))
    strategy2 = ((4, 1), (1, 1))
    with no_init_parameters():
        net = Pipeline(PipelineSplit(strategy1, strategy2), 4, stage_config={"cell.block.0": 0, "cell.block.1": 1})
        params = net.network.cell.block[1].trainable_params()
        optimizer = nn.Lamb(params, learning_rate=0.01)
    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
    parallel_net.pipeline(stages=2)
    dataset = DatasetLenet(data, label, 3)
    model = Model(parallel_net, optimizer=optimizer)
    model.train(2, dataset, dataset_sink_mode=False)
    group_info_list = restore_group_info_list("./test_pipeline_split_stage1_mirror_group.pb")
    assert group_info_list == [39, 47, 55, 63]
    del os.environ['GROUP_INFO_FILE']
