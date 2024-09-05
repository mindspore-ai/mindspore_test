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

import os
import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor, Parameter
from mindspore._c_expression import send_recv
from mindspore.communication import init

class SendNet(nn.Cell):
    def __init__(self):
        super(SendNet, self).__init__()
        self.depend = ops.Depend()
        self.send = ops.Send(sr_tag=0, dest_rank=1, group="hccl_world_group")
        self.param = Parameter(Tensor(np.ones([2, 8], dtype=np.float32) * 3), name='param_send')

    def construct(self, x):
        out = self.depend(x + self.param, self.send(x))
        return out

class ReceiveNet(nn.Cell):
    def __init__(self):
        super(ReceiveNet, self).__init__()
        self.recv = ops.Receive(sr_tag=0, src_rank=0, shape=[2, 8], dtype=ms.float32, group="hccl_world_group")
        self.param = Parameter(Tensor(np.ones([2, 8], dtype=np.float32) * 2), name='param_recv')

    def construct(self):
        out = self.recv()
        return out + self.param

init()
ms.set_context(mode=ms.GRAPH_MODE, jit_level='O2')

if __name__ == '__main__':
    rank_id = os.environ['RANK_ID']
    rank_size = os.environ['RANK_SIZE']

    print(f'rank_id={rank_id}/{rank_size}')

    if rank_id == '0':
        input_x = Tensor(np.arange(0, 16, dtype=np.float32).reshape(2, 8))
        net_send = SendNet()
        output_send = net_send(input_x)
        print(output_send.asnumpy().shape)
        send_recv([input_x, net_send.param], src_rank=1, dst_rank=0)
        print('send net param:', net_send.param.value())
    else:
        net_recv = ReceiveNet()
        output_recv = net_recv()
        print(output_recv.asnumpy().shape)
        send_recv([output_recv, net_recv.param], src_rank=1, dst_rank=0)
        print('recv net param:', net_recv.param.value())
