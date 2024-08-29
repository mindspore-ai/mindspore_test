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

import numpy as np
import mindspore as ms
import mindspore.ops.operations as P
from mindspore.nn import Cell
from mindspore import context, Tensor, Parameter
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from parallel.utils.utils import compile_net

def generate_inputs(B, N1, N2, S1, S2, D, mask_shape, keep_prob=0, input_layout="BSH",
                    dtype=ms.float16, attn_mask=None):
    if input_layout == "BSH":
        query = Tensor(np.ones((B, S1, N1 * D)), dtype=dtype)
        key = Tensor(np.ones((B, S2, N2 * D)), dtype=dtype)
        value = Tensor(np.ones((B, S2, N2 * D)), dtype=dtype)
    elif input_layout == "BNSD":
        query = Tensor(np.ones((B, N1, S1, D)), dtype=dtype)
        key = Tensor(np.ones((B, N2, S2, D)), dtype=dtype)
        value = Tensor(np.ones((B, N2, S2, D)), dtype=dtype)
    elif input_layout == "SBH":
        query = Tensor(np.ones((S1, B, N1 * D)), dtype=dtype)
        key = Tensor(np.ones((S2, B, N2 * D)), dtype=dtype)
        value = Tensor(np.ones((S2, B, N2 * D)), dtype=dtype)
    else:
        raise ValueError(f"input_layout not valid")
    if keep_prob < 1.0:
        drop_rate = 1.0 - keep_prob
        drop_mask = np.random.choice([0, 1], size=mask_shape, p=[drop_rate, keep_prob]).astype(np.uint8)
        drop_mask_bits = np.packbits(drop_mask, bitorder="little").view(np.uint8).reshape(
            mask_shape[0:-1]+(mask_shape[-1] // 8,))
    else:
        drop_mask_bits = None
    if keep_prob >= 1.0:
        if attn_mask is not None:
            return query, key, value, attn_mask
        return query, key, value
    return query, key, value, Tensor(drop_mask_bits)


class FlashAttentionScoreNet(Cell):
    def __init__(self, head_num, mul_size1, mul_size2, keep_prob=0.9, input_layout="BSH", sparse_mode=0, strategy=None):
        super(FlashAttentionScoreNet, self).__init__()
        self.reshape = P.Reshape()
        self.drop_gen_mask = P.DropoutGenMask()
        self.keep_prob = Tensor(keep_prob, ms.float16)
        self.head_num = head_num
        self.input_layout = input_layout
        self.fa_op = FlashAttentionScore(head_num=head_num, keep_prob=keep_prob, sparse_mode=sparse_mode,
                                         input_layout=input_layout)
        self.add = P.Add()
        mul_np1 = np.full(mul_size1, 0.1, dtype=np.float16)
        mul_np2 = np.full(mul_size2, 0.1, dtype=np.float16)
        self.mul_weight1 = Parameter(Tensor(mul_np1), name="mul_weight1")
        self.mul_weight2 = Parameter(Tensor(mul_np2), name="mul_weight2")
        self.mul = P.Mul()
        if strategy is not None:
            self.fa_op.shard(strategy)

    def construct(self, query, key, value, drop_mask_bits=None, attn_mask=None):
        drop_mask_bits = ms.ops.stop_gradient(drop_mask_bits)
        attn_mask = ms.ops.stop_gradient(attn_mask)
        query = self.mul(query, self.mul_weight1)
        key = self.mul(key, self.mul_weight2)
        _, _, _, out = self.fa_op(query, key, value, None, drop_mask_bits, None, attn_mask, None)
        return out


def test_ring_semi_auto_bsh_cut_not_full():
    """
    Feature: test ring attention semi auto parallel
    Description: bsh with strategy < num_dev
    Expectation: compile success.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    B, N1, N2, S1, S2, D = 1, 32, 16, 4096, 4096, 128
    input_layout = "BSH"
    data = generate_inputs(B, N1, N2, S1, S2, D, mask_shape=(B, N1, S1, S2), keep_prob=1.0, input_layout=input_layout)

    net = FlashAttentionScoreNet(N1, input_layout=input_layout, mul_size1=(B, S1, N1*D), mul_size2=(B, S2, N2*D),
                                 strategy=((1, 2, 1), (1, 2, 1), (1, 2, 1)), keep_prob=1.0)
    net.fa_op.add_prim_attr("enable_ring_attention", True)
    net.fa_op.add_prim_attr("enable_ra_send_recv", True)
    compile_net(net, *data)

def test_ring_semi_auto_bnsh_cut_not_full():
    """
    Feature: test ring attention semi auto parallel
    Description: bnsd with strategy < num_dev
    Expectation: compile success.
    """
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    B, N1, N2, S1, S2, D = 1, 32, 16, 4096, 4096, 128
    input_layout = "BNSD"
    data = generate_inputs(B, N1, N2, S1, S2, D, mask_shape=(B, N1, S1, S2), keep_prob=1.0, input_layout=input_layout)
    net = FlashAttentionScoreNet(N1, input_layout=input_layout, mul_size1=(B, N1, S1, D), mul_size2=(B, N2, S2, D),
                                 strategy=((1, 1, 2, 1), (1, 1, 2, 1), (1, 1, 2, 1)), keep_prob=1.0)
    net.fa_op.add_prim_attr("enable_ring_attention", True)
    net.fa_op.add_prim_attr("enable_ra_send_recv", True)
    compile_net(net, *data)
