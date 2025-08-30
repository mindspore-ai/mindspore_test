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
import mindspore.runtime as rt
from mindspore import Layout
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

def double_grad(x):
    return x * 2

def bprop_fn(x, rank_id, out, dout):
    return (double_grad(dout), double_grad(dout))

class MorpTestNet(nn.Cell):
    def __init__(self, rank_id, dp):
        super(MorpTestNet, self).__init__()
        self.rank_id = rank_id
        self.dp = dp
        self.add = ops.Add().shard(((self.dp,), (self.dp,)))
        self.layout = Layout((self.dp,), ("dp",))
        self.morph = ops.Morph(all2allv, infer_shape, infer_dtype, bprop_fn=bprop_fn)
        self.morph.add_prim_attr("self_define_shard", True)
        self.morph.shard(
            in_strategy=(self.layout("dp",),),
            out_strategy=(self.layout("dp",),)
        )

    def construct(self, x1, x2):
        o1 = self.add(x1, x2)
        o2 = self.morph(o1, self.rank_id)
        o3 = self.add(o2, x2)
        return o3

def init_env():
    init()
    context.set_context(mode=ms.GRAPH_MODE, device_target='Ascend', jit_level="O1")
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel")
    context.set_auto_parallel_context(dataset_strategy="full_batch")

def test_semi_auto_parallel():
    rt.launch_blocking()
    init_env()

    x1 = ms.Tensor(np.arange(1, 9), dtype=ms.float32)
    x2 = ms.Tensor(np.arange(1, 9) * 0.1, dtype=ms.float32)

    rank_id = get_rank()
    dp = get_group_size()

    net = MorpTestNet(rank_id, dp)
    grad_op = ops.GradOperation(get_all=True)
    grad_net = grad_op(net)

    grad = grad_net(x1, x2)
    dx1 = grad[0].asnumpy()
    dx2 = grad[1].asnumpy()

    if rank_id == 0:
        rank0_dx1 = np.array([1] * 4 + [0] * 4, dtype=np.float32)
        rank0_dx2 = np.array(list(map(double_grad, [2] * 4)) + [0] * 4, dtype=np.float32)
        np.allclose(dx1, rank0_dx1)
        np.allclose(dx2, rank0_dx2)
    else:
        rank1_dx1 = np.array([0] * 4 + [1] * 4, dtype=np.float32)
        rank1_dx2 = np.array([0] * 4 + list(map(double_grad, [2] * 4)), dtype=np.float32)
        np.allclose(dx1, rank1_dx1)
        np.allclose(dx2, rank1_dx2)
