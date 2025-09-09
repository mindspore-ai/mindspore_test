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
import os
import pytest
from tests.mark_utils import arg_mark
from mindspore import nn, ops
from mindspore.common.parameter import Parameter
from mindspore.common.initializer import initializer
from mindspore.communication.management import init
from mindspore.nn.utils import no_init_parameters
from mindspore.parallel import init_parameters
from mindspore.parallel.spmd.hsdp.hsdp import hsdp

class NetL2(nn.Cell):
    def __init__(self, in_channels, hidden_size):
        super(NetL2, self).__init__()
        self.dense1 = nn.Dense(in_channels, hidden_size, weight_init="ones", has_bias=False)
        self.bias = Parameter(initializer("zeros", [hidden_size], self.dense1.weight.dtype))
        self.add = ops.BiasAdd()

    def construct(self, x):
        x = self.dense1(x)
        x = self.add(x, self.bias)
        return x

class NetL3(nn.Cell):
    def __init__(self, in_channels, out_channels, hidden_size):
        super(NetL3, self).__init__()
        self.block = NetL2(in_channels, hidden_size)
        self.dense2 = nn.Dense(hidden_size, out_channels, weight_init="ones", has_bias=False)

    def construct(self, x):
        x = self.block(x)
        x = self.dense2(x)
        return x

@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
@pytest.mark.parametrize('use_hsdp', [True, False])
def test_init_parameters(use_hsdp):
    '''
    Feature: init parameters.
    Description: test init parameter interface.
    Expectation: Run success
    '''
    os.environ["MS_SIMULATION_LEVEL"] = "0"
    os.environ["RANK_SIZE"] = "32"
    os.environ["RANK_ID"] = "0"
    init()

    in_channels = 128
    out_channels = 32
    hidden_size = 512
    with no_init_parameters():
        net = NetL3(in_channels, out_channels, hidden_size)
    if use_hsdp:
        shard_size = 4
        threshold = 4
        optimizer_level = "level1"
        hsdp(net, shard_size, threshold, optimizer_level)

    init_parameters(net)
