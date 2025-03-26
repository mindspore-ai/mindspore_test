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

import mindspore as ms
from mindspore import nn

from mindspore.parallel.nn.parallel_grad_reducer import PipelineGradReducer as NewPipelineGradReducer
from mindspore.parallel.nn.parallel_cell_wrapper import PipelineCell as NewPipelineCell
from mindspore.parallel.nn import MicroBatchInterleaved as NewMicroBatchInterleaved

from mindspore.nn.wrap.grad_reducer import PipelineGradReducer
from mindspore.nn.wrap.cell_wrapper import PipelineCell
from mindspore.nn import MicroBatchInterleaved

# pylint: disable=unused-import
from mindspore.parallel import load_distributed_checkpoint as new_load_distributed_checkpoint
from mindspore.parallel import merge_sliced_parameter as new_merge_sliced_parameter
from mindspore.parallel import build_searched_strategy as new_build_searched_strategy
from mindspore.parallel import restore_group_info_list as new_restore_group_info_list

from mindspore.train.serialization import load_distributed_checkpoint
from mindspore.train.serialization import merge_sliced_parameter
from mindspore.train.serialization import build_searched_strategy
from mindspore.train.serialization import restore_group_info_list

from mindspore.train import load_distributed_checkpoint as second_load_distributed_checkpoint
from mindspore.train import merge_sliced_parameter as second_merge_sliced_parameter
from mindspore.train import build_searched_strategy as second_build_searched_strategy
from mindspore.train import restore_group_info_list as second_restore_group_info_list
from mindspore.nn.utils import no_init_parameters


ms.set_context(mode=ms.GRAPH_MODE)
ms.set_seed(1)


def test_pipeline_grad_reducer():
    """
    Feature: test class PipelineGradReducer, PipelineCell, MicroBatchInterleaved.
    Description: fixed compatibility issues caused by interface relocation.
    Expectation: success
    """
    class Network(nn.Cell):
        def __init__(self, in_features, out_features, sens=1.0):
            super().__init__()
            self.layer1 = nn.Dense(in_features, 16)
            self.relu1 = nn.ReLU()
            self.layer2 = nn.Dense(16, 16)
            self.relu2 = nn.ReLU()
            self.layer3 = nn.Dense(16, out_features)

        def construct(self, x):
            x = self.layer1(x)
            x = self.relu1(x)
            x = self.layer2(x)
            x = self.relu2(x)
            logits = self.layer3(x)
            return logits

    in_features, out_features = 32, 10
    with no_init_parameters():
        net = Network(in_features, out_features)
    net.layer1.pipeline_stage = 0
    net.relu1.pipeline_stage = 0
    net.layer2.pipeline_stage = 0
    net.relu2.pipeline_stage = 1
    net.layer3.pipeline_stage = 1
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.SGD(net.trainable_params(), 1e-2)

    _ = NewPipelineCell(nn.WithLossCell(net, loss_fn), 2)
    _ = PipelineCell(nn.WithLossCell(net, loss_fn), 2)

    _ = NewPipelineGradReducer(optimizer.parameters)
    _ = PipelineGradReducer(optimizer.parameters)

    _ = NewMicroBatchInterleaved(net, 2)
    _ = MicroBatchInterleaved(net, 2)
