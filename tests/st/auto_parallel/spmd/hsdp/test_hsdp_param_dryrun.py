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
from mindspore import nn
from mindspore.parallel import Layout
from mindspore.nn.utils import no_init_parameters
from mindspore.communication.management import init
from mindspore.parallel.spmd.hsdp.hsdp_param import HSDPParam
from mindspore.parallel.spmd.hsdp.hsdp_comm import HSDPComm
from mindspore.parallel.spmd.hsdp.hsdp_utils import OptimizerLevel, HSDPConfig
os.environ["MS_SIMULATION_LEVEL"] = "0"
os.environ["RANK_SIZE"] = "32"
os.environ["RANK_ID"] = "0"
init()

def hsdp_param_to_unsharded(net):
    shard_size = 2
    threshold = 1
    requires_acc_grad = True
    shard_level = OptimizerLevel.SHARD_OPT
    use_cell_hook = True
    comm = HSDPComm()
    hsdp_config = HSDPConfig(shard_size, threshold, requires_acc_grad, shard_level, use_cell_hook)
    hsdp_param = HSDPParam(net, net.weight.name, net.weight, comm, hsdp_config)
    hsdp_param.to_sharded()
    hsdp_param.to_unsharded()
    hsdp_param.zero_acc_grad()

def test_hsdp_param_to_unsharded():
    """
    Feature: hsdp param.
    Description: change hsdp param to unshared state.
    Expectation: change hsdp param to unshared state without error.
    """
    in_channels = 256
    out_channels = 64
    net = nn.Dense(in_channels, out_channels, weight_init="ones")
    hsdp_param_to_unsharded(net)

def test_hsdp_no_init_param_to_unsharded():
    """
    Feature: hsdp not init param.
    Description: change hsdp param to unshared state.
    Expectation: change hsdp param to unshared state without error.
    """
    in_channels = 256
    out_channels = 64
    with no_init_parameters():
        net = nn.Dense(in_channels, out_channels, weight_init="ones")
    hsdp_param_to_unsharded(net)

def test_hsdp_param_with_layout():
    """
    Feature: hsdp param with layout.
    Description: construct and init hsdp param with layout.
    Expectation: construct hsdp param without error.
    """
    in_channels = 256
    out_channels = 64
    net = nn.Dense(in_channels, out_channels, weight_init="ones")

    device_matrix = (4, 8)
    alias_name = ("dp", "mp")
    rank_list = list(range(32))
    layout = Layout(device_matrix, alias_name, rank_list)
    w_layout = layout("mp", "None")
    net.weight = net.weight.local_to_global(w_layout)

    hsdp_param_to_unsharded(net)

def test_hsdp_no_init_param_with_layout():
    """
    Feature: hsdp no init param with layout.
    Description: construct and init hsdp param with layout.
    Expectation: construct hsdp param without error.
    """
    in_channels = 256
    out_channels = 64
    with no_init_parameters():
        net = nn.Dense(in_channels, out_channels, weight_init="ones")

    device_matrix = (4, 8)
    alias_name = ("dp", "mp")
    rank_list = list(range(32))
    layout = Layout(device_matrix, alias_name, rank_list)
    w_layout = layout("mp", "None")
    net.weight = net.weight.local_to_global(w_layout)

    hsdp_param_to_unsharded(net)
