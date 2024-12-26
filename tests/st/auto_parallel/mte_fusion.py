# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import mindspore.communication.management as D
from mindspore import context, Tensor

grad = C.GradOperation(get_all=True)


class AllGatherMatmulNet(nn.Cell):
    def __init__(self, seq_len, hidden_size, dp, mp):
        super(AllGatherMatmulNet, self).__init__()
        self.gelu1 = P.Gelu()
        self.dense1 = nn.Dense(in_channels=hidden_size,
                               out_channels=hidden_size,
                               weight_init="ones").to_float(mstype.float16)
        self.gelu2 = P.Gelu()
        self.gelu1.shard(((dp * mp, 1),))
        self.dense1.matmul.shard(((dp, 1), (mp, 1)))
        self.dense1.bias_add.shard(((dp, mp), (mp,)))
        self.gelu2.shard(((dp, mp),))
        self.dense2.matmul.shard(((dp, 1), (mp, 1)))

    def construct(self, x):
        out = self.gelu1(x)
        out = self.dense1(out)
        out1 = self.gelu1(x)
        out = self.dense1(out1)
        out = self.gelu2(out)
        out1 = self.dense2(out1)
        return out1, out


class MatmulReduceScatterNet(nn.Cell):
    def __init__(self, seq_len, hidden_size, dp, mp):
        super(MatmulReduceScatterNet, self).__init__()
        self.dense1 = nn.Dense(in_channels=hidden_size,
                               out_channels=hidden_size,
                               weight_init="ones").to_float(mstype.float16)
        self.gelu1 = P.Gelu()
        self.gelu2 = P.Gelu()

        self.gelu1.shard((((dp, 1),)))
        self.dense1.matmul.shard(((dp, mp), (1, mp)), out_strategy=((dp * mp, 1),))
        self.dense1.bias_add.shard(((dp * mp, 1), (1,)))
        self.gelu2.shard(((dp, mp),))

    def construct(self, x):
        out = self.gelu1(x)
        out = self.dense1(out)
        out = self.gelu2(out)
        return out


def test_all_gather_matmul_forward():
    '''
    Feature: MTE fusion.
    Description: Test all_gather-matmul fusion in forward.
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_config={"jit_level": "O0"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
    D.init()
    seq_len, hidden_size = 2048, 4096
    dp, mp = 1, 8
    x = Tensor(np.random.uniform(-3, 3, [seq_len, hidden_size]), dtype=mstype.float16)

    os.environ['MS_ENABLE_LCCL'] = "off" #关闭lccl纯通信
    os.environ['LCCL_ENABLE_FUSION'] = "off" #关闭lccl通算

    net = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    expect_out1 = net(x)[0].asnumpy()
    expect_out = net(x)[1].asnumpy()

    os.environ['MS_ENABLE_LCCL'] = "on" #使能lccl纯通信
    os.environ['LCCL_ENABLE_FUSION'] = "on" #使能lccl通算

    mte_net = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    mte_out1 = mte_net(x)[0].asnumpy()
    mte_out = mte_net(x)[1].asnumpy()

    assert np.allclose(expect_out1, mte_out1, 1e-2, 1e-2)
    assert np.allclose(expect_out, mte_out, 1e-2, 1e-2)


def test_matmul_reduce_scatter_forward():
    '''
    Feature: MTE fusion.
    Description: Test matmul-reduce_scatter fusion in forward.
    Expectation: Run success
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_config={"jit_level": "O0"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
    D.init()
    seq_len, hidden_size = 4096, 12288
    dp, mp = 1, 8
    x = Tensor(np.random.uniform(-3, 3, [seq_len, hidden_size]), dtype=mstype.float16)

    os.environ['MS_ENABLE_LCCL'] = "off" #关闭lccl纯通信
    os.environ['LCCL_ENABLE_FUSION'] = "off" #关闭lccl通算

    net = MatmulReduceScatterNet(seq_len, hidden_size, dp, mp)
    expect_out = net(x).asnumpy()

    os.environ['MS_ENABLE_LCCL'] = "on" #使能lccl纯通信
    os.environ['LCCL_ENABLE_FUSION'] = "on" #使能lccl通算

    mte_net = MatmulReduceScatterNet(seq_len, hidden_size, dp, mp)
    mte_out = mte_net(x).asnumpy()

    assert np.allclose(expect_out, mte_out, 1e-2, 1e-2)
