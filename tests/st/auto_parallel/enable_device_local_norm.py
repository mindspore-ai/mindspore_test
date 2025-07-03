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

from mindspore import Tensor, context
from mindspore.communication import init, get_rank
from mindspore.nn import Momentum, TrainOneStepCell
from mindspore.ops._grad_experimental.grad_comm_ops import get_squared_device_local_norm_param
from mindspore.parallel.auto_parallel import AutoParallel
from .dump_local_norm import Net, calc_squared_device_local_norm

context.set_context(mode=context.GRAPH_MODE)


def test_enable_device_local_norm():
    """
    Feature: Test dump local norm using AutoParallel.enable_device_local_norm()
    Description: Test dump local norm using AutoParallel.enable_device_local_norm()
    Expectation: local norms match with device local norm on each device.
    """
    init()
    dump_path = "./dump_path"
    net = Net()
    optimizer = Momentum(net.trainable_params(), learning_rate=0.1, momentum=0.9)
    net = TrainOneStepCell(net, optimizer)
    net.set_train()

    parallel_net = AutoParallel(net, parallel_mode="semi_auto")
    parallel_net.dataset_strategy = "full_batch"
    parallel_net.dump_local_norm(dump_path)
    parallel_net.enable_device_local_norm()
    x = Tensor(np.random.rand(8, 1).astype(np.float32))
    parallel_net(x)

    rank_dump_path = dump_path + "/rank_" + str(get_rank())
    squared_norm1 = calc_squared_device_local_norm(rank_dump_path)
    squared_norm2 = get_squared_device_local_norm_param().asnumpy()
    assert np.isclose(squared_norm1, squared_norm2)
