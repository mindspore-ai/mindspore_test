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

"""test hccl get_comm_name with 8p"""

import re
import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.communication.management import init, get_comm_name, create_group, get_rank
from mindspore import context
from mindspore.communication import GlobalComm

np.random.seed(1)
context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
init()
user_group = "test_comm_name"
rank_ids = [0, 7]
if get_rank() in rank_ids:
    create_group(user_group, rank_ids)

def check_comm_name(text):
    comm_name_pattern = r"\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}%[a-zA-Z0-9]+(_\d+)+"
    assert re.match(comm_name_pattern, text), f"{text} is not comm_name!"


class GroupCommNameNet(nn.Cell):
    def __init__(self, group=GlobalComm.WORLD_COMM_GROUP):
        super(GroupCommNameNet, self).__init__()
        self.group = group

    def construct(self, x):
        return get_comm_name()


def test_hccl_get_comm_name_func_net_8p():
    """
    Feature: test 'get_comm_name' communication function in cell.
    Description: test 'get_comm_name' communication function in cell.
    Expectation: expect correct result.
    """
    net = GroupCommNameNet()
    test_tensor = np.ones([3, 4]).astype(np.float32)
    output = net(Tensor(test_tensor, mstype.float32))
    check_comm_name(output)
    print("Get world group comm_name: ", output)


def test_hccl_get_comm_name_func_8p():
    """
    Feature: test 'get_comm_name' communication function.
    Description: test 'get_comm_name' communication function.
    Expectation: expect correct result.
    """
    output_world = get_comm_name()
    check_comm_name(output_world)
    if get_rank() in rank_ids:
        output_other = get_comm_name(user_group)
        check_comm_name(output_other)
        print("Get world group comm_name: ", output_world, "created group comm_name: ", output_other)
    else:
        print("Get world group comm_name: ", output_world)
