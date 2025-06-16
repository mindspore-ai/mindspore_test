# Copyright 2023 Huawei Technologies Co., Ltd
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
import argparse

import mindspore.context as context
import mindspore.dataset as ds
import mindspore.dataset.transforms as C
import mindspore.dataset.vision as CV
import mindspore.nn as nn
import numpy as np
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.dataset.vision import Inter
from mindspore.train import Model, LossMonitor, Accuracy
from mindspore.common.initializer import TruncatedNormal
from mindspore.communication import init, get_rank, get_group_size, create_group
import mindspore.communication.comm_func as comm_func
from mindspore.mint.distributed.distributed import barrier
from mindspore.mint.distributed.distributed import (
    init_process_group,
    get_rank,
    get_world_size,
    get_backend,
    new_group,
    get_global_rank,
    get_process_group_ranks,
    broadcast,
    gather,
    scatter,
    all_gather,
    send,
    recv,
    barrier,
    all_reduce,
)

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')

init()

rank = get_rank()
size = get_group_size()

mccl_rank_list = list(range(0, size))
print(f"Start create inter-node group mccl_inter_node with {mccl_rank_list}")
create_group("mccl_inter_node", mccl_rank_list)

hccl_rank_list1 = list(range(0, size // 2))
hccl_rank_list2 = list(range(size // 2, size))
if rank in hccl_rank_list1:
    print(f"Start create intar-node group hccl_intra_node1 with {hccl_rank_list1}")
    create_group("hccl_intra_node1", hccl_rank_list1)
if rank in hccl_rank_list2:
    print(f"Start create intar-node group hccl_intra_node2 with {hccl_rank_list2}")
    create_group("hccl_intra_node2", hccl_rank_list2)



class AllReduceNet(nn.Cell):
    def __init__(self):
        super(AllReduceNet, self).__init__()
        self.mul = P.Mul()
        if rank in hccl_rank_list1:
            self.all_reduce = P.AllReduce(group="hccl_intra_node1")
        if rank in hccl_rank_list2:
            self.all_reduce = P.AllReduce(group="hccl_intra_node2")
        self.add = P.Add()
        self.y1 = Tensor(np.array([[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]])).astype(np.float32)
        self.y2 = Tensor(np.array([[-16, -16, -16, -16], [-16, -16, -16, -16], \
                                   [-16, -16, -16, -16]])).astype(np.float32)

    def construct(self, x):
        x = self.mul(x, 2)
        z = self.add(x, self.y1)
        z = self.add(z, self.y1)
        z = self.all_reduce(z)
        out = self.add(z, self.y2)
        out = self.all_reduce(out)
        out = self.mul(out, 2)
        return out

net = AllReduceNet()
input_x = np.ones([3, 4]).astype(np.float32)
output = net(Tensor(input_x, mstype.float32))

input_tensor = output
recv_output = ms.Tensor(np.zeros([3, 4]).astype(np.float32))
if rank // (size // 2) == 0:
    print(f"send rank is {rank} to {rank + size // 2} {input_tensor * rank}", flush=True)
    send(input_tensor * rank, rank + size // 2, group="mccl_inter_node")
else:
    out = recv(recv_output, src=rank - size // 2, group="mccl_inter_node")
    print(f"send rank is {rank} from {rank - size // 2} {recv_output.asnumpy()}", flush=True)