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
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.communication import get_rank, init, get_group_size

def infer_dtype(*args):
    return args[0]

def infer_shape(*args):
    return args[0]

def all2allv(x, rank_id):
    if rank_id == 0:
        send_list, recv_list = [1, 3], [1, 2]
    else:
        send_list, recv_list = [2, 2], [3, 2]

    x = ops.AlltoAllV()(x, send_list, recv_list)
    x = ops.AlltoAllV()(x, recv_list, send_list)

    return x

class MorpTestNet(nn.Cell):
    def __init__(self):
        super(MorpTestNet, self).__init__()
        self.rank_id = get_rank()
        self.dp = get_group_size()
        self.add = ops.Add().shard(((self.dp,), (self.dp,)))
        self.morph = ops.Morph(all2allv, infer_shape, infer_dtype)

    def construct(self, x1, x2):
        o1 = self.add(x1, x2)
        o2 = self.morph(o1, self.rank_id)
        o3 = self.add(o2, x2)
        return o3

def init_env():
    init()
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', jit_level="O1")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")

def test_semi_auto_parallel():
    init_env()

    x1 = ms.Tensor(np.arange(1, 9), dtype=ms.float32)
    x2 = ms.Tensor(np.arange(1, 9) * 0.1, dtype=ms.float32)

    net = MorpTestNet()
    net.shard(((get_group_size(),),))
    out = net(x1, x2)
    print("out: ", out)
