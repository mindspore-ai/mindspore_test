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
import mindspore as ms
from mindspore import ops, nn, context, Tensor
from mindspore.communication.management import init
from mindspore.communication._comm_helper import _get_group_map
from mindspore._c_expression import _rebuild_world_group, _rebuild_sub_group, _finalize_comm


class AllReduceNet(nn.Cell):
    def __init__(self):
        super(AllReduceNet, self).__init__()
        self.allreduce = ops.AllReduce()

    def construct(self, x):
        output = self.allreduce(x)
        return output


def rebuild_hccl_interface():
    """
    Feature: test hcom destroy and rebuild.
    Description: test graceful exit, save ckpt after exit training process.
    Expectation: none.
    """
    # init
    group_name = "hccl_world_group"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", ascend_config={"hccl_watchdog": False})
    init()
    print("==INIT OK==")
    base_value = Tensor([8], ms.int32)
    t1 = Tensor([1], ms.int32)
    net = AllReduceNet()
    output1 = net(t1)
    group_map = _get_group_map()
    assert len(group_map) == 1
    assert group_name in group_map.keys()
    assert output1.asnumpy() == base_value.asnumpy()

    # destroy hcom
    _finalize_comm()
    # rebuild group
    _rebuild_world_group()
    _rebuild_sub_group()
    output2 = net(t1)
    assert output2.asnumpy() == base_value.asnumpy()


rebuild_hccl_interface()
