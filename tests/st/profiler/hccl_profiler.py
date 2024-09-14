# Copyright 2024 Huawei Technologies Co., Ltd
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

"""test hccl allreduce with 2p profiling"""

import mindspore.context as context
from mindspore import Profiler
from mindspore.communication.management import init, get_rank, get_group_size
from model_zoo import AllReduceNet

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
context.set_context(jit_level='O0')

init()
rank = get_rank()
size = get_group_size()

def test_AllReduce():
    """
    Feature: hccl operator test.
    Description: msrun hccl all_reduce 2P case.
    Expectation: success
    """
    profiler = Profiler(output_path="profiler_hccl_data_" + str(rank), profile_communication=True,
                        data_simplification=False)
    all_reduce = AllReduceNet()
    all_reduce()
    profiler.analyse()
