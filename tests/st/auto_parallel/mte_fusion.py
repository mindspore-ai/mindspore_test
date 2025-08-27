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
import subprocess
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore.ops import composite as C
from mindspore.ops import operations as P
import mindspore.communication.management as D
from mindspore import context, Tensor

grad = C.GradOperation(get_all=True)


def check_keyword_in_ir(ir_path, kernel, keyword):
    cmd = f"grep '= {kernel}' {ir_path}"
    print(f'===== cmd: {cmd}', flush=True)
    result = subprocess.getoutput(cmd)
    return result.find(keyword) != -1


class AllGatherMatmulNet(nn.Cell):
    def __init__(self, seq_len, hidden_size, dp, mp):
        super(AllGatherMatmulNet, self).__init__()
        self.gelu1 = P.GeLU()
        self.dense1 = nn.Dense(in_channels=hidden_size,
                               out_channels=hidden_size,
                               weight_init="ones").to_float(mstype.float16)
        self.gelu2 = P.GeLU()
        self.dense2 = nn.Dense(in_channels=hidden_size,
                               out_channels=hidden_size,
                               weight_init="ones").to_float(mstype.float16)
        self.gelu1.shard(((dp * mp, 1),))
        self.dense1.matmul.shard(((dp, 1), (mp, 1)))
        self.dense1.bias_add.shard(((dp, mp), (mp,)))
        self.gelu2.shard(((dp, mp),))
        self.dense2.matmul.shard(((dp, 1), (mp, 1)))

    def construct(self, x):
        out = self.gelu1(x)
        out1 = self.dense2(out)
        out = self.dense1(out)
        out = self.gelu2(out)
        return out1, out


class MatmulReduceScatterNet(nn.Cell):
    def __init__(self, seq_len, hidden_size, dp, mp):
        super(MatmulReduceScatterNet, self).__init__()
        self.dense1 = nn.Dense(in_channels=hidden_size,
                               out_channels=hidden_size,
                               weight_init="ones").to_float(mstype.float16)
        self.gelu1 = P.GeLU()
        self.gelu2 = P.GeLU()

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
    os.environ['MS_ENABLE_LCCL'] = "on"
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_config={"jit_level": "O0"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
    D.init()
    seq_len, hidden_size = 4096, 12288
    dp, mp = 1, 4
    x = Tensor(np.random.uniform(-3, 3, [seq_len, hidden_size]), dtype=mstype.float16)
    context.set_context(compute_communicate_fusion_level=0)
    net = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    expect_out1 = net(x)[0].asnumpy()
    expect_out = net(x)[1].asnumpy()

    context.set_context(compute_communicate_fusion_level=3)
    mte_net = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    mte_out1 = mte_net(x)[0].asnumpy()
    mte_out = mte_net(x)[1].asnumpy()

    print("====================================================="
          f"\n expect_out1:\n{expect_out1}, \n expect_output2:\n{expect_out},"
          f"\n mte_out1:\n{mte_out1}, \n mte_out2:\n{mte_out},", flush=True)
    assert np.allclose(expect_out1, mte_out1, 1e-2, 1e-2)
    assert np.allclose(expect_out, mte_out, 1e-2, 1e-2)


def test_matmul_reduce_scatter_forward():
    '''
    Feature: MTE fusion.
    Description: Test matmul-reduce_scatter fusion in forward.
    Expectation: Run success
    '''
    os.environ['MS_ENABLE_LCCL'] = "on"
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_config={"jit_level": "O0"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
    D.init()
    seq_len, hidden_size = 4096, 12288
    dp, mp = 1, 4
    x = Tensor(np.random.uniform(-3, 3, [seq_len, hidden_size]), dtype=mstype.float16)
    context.set_context(compute_communicate_fusion_level=0)
    net = MatmulReduceScatterNet(seq_len, hidden_size, dp, mp)
    expect_out = net(x).asnumpy()

    context.set_context(compute_communicate_fusion_level=3)
    mte_net = MatmulReduceScatterNet(seq_len, hidden_size, dp, mp)
    mte_out = mte_net(x).asnumpy()

    print("====================================================="
          f"\n expect_out:\n{expect_out}, \n mte_out:\n{mte_out},", flush=True)
    assert np.allclose(expect_out, mte_out, 1e-2, 1e-2)


def test_ms_disable_lccl_kernels_list():
    """
    Feature: MTE fusion.
    Description: Test environment MS_DISABLE_KERNELS_LIST.
    Expectation: Run success
    """
    os.environ['MS_ENABLE_LCCL'] = "on"
    os.environ['MS_DEV_SAVE_GRAPHS'] = str(2)
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = './graph_lccl_kernels_list'
    os.environ['MS_DEV_DUMP_IR_PASSES'] = 'graph_build'
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')
    context.set_context(jit_config={"jit_level": "O0"})
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", dataset_strategy="full_batch")
    context.set_context(compute_communicate_fusion_level=3)
    D.init()
    seq_len, hidden_size = 4096, 12288
    dp, mp = 1, 4
    x = Tensor(np.random.uniform(-3, 3, [seq_len, hidden_size]), dtype=mstype.float16)

    kernel_allgather = 'AllGather'
    kernel_allgathermatmul = 'PrimFunc_AllGatherMatmul'
    keyword_lccl = 'collective_comm_lib: "LCCL"'
    keyword_hccl = 'collective_comm_lib: "HCCL"'

    # Unset black list, AllGather go lccl, AllGatherMatmul go lccl.
    net0 = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    out0 = net0(x)[0].asnumpy()
    out1 = net0(x)[1].asnumpy()
    print("====================================================="
          f"\n output0:\n{out0}, \n output1:\n{out1},", flush=True)
    path = './graph_lccl_kernels_list/rank_0/graph_build_0_*'
    assert check_keyword_in_ir(path, kernel_allgather, keyword_lccl)
    assert not check_keyword_in_ir(path, kernel_allgather, keyword_hccl)
    assert check_keyword_in_ir(path, kernel_allgathermatmul, keyword_lccl)
    assert not check_keyword_in_ir(path, kernel_allgathermatmul, keyword_hccl)
    print(f'===== end graph_build_0 +++', flush=True)

    # Blank black list, AllGather go lccl, AllGatherMatmul go lccl.
    os.environ['MS_DISABLE_LCCL_KERNELS_LIST'] = ""
    net1 = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    out0 = net1(x)[0].asnumpy()
    out1 = net1(x)[1].asnumpy()
    print("====================================================="
          f"\n output0:\n{out0}, \n output1:\n{out1},", flush=True)
    path = './graph_lccl_kernels_list/rank_0/graph_build_1_*'
    assert check_keyword_in_ir(path, kernel_allgather, keyword_lccl)
    assert not check_keyword_in_ir(path, kernel_allgather, keyword_hccl)
    assert check_keyword_in_ir(path, kernel_allgathermatmul, keyword_lccl)
    assert not check_keyword_in_ir(path, kernel_allgathermatmul, keyword_hccl)
    print(f'===== end graph_build_1 +++', flush=True)

    # Black list for only 'AllGather', AllGather go hccl, AllGatherMatmul go lccl.
    os.environ['MS_DISABLE_LCCL_KERNELS_LIST'] = "AllGather,"
    net2 = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    out0 = net2(x)[0].asnumpy()
    out1 = net2(x)[1].asnumpy()
    print("====================================================="
          f"\n output0:\n{out0}, \n output1:\n{out1},", flush=True)
    path = './graph_lccl_kernels_list/rank_0/graph_build_2_*'
    assert not check_keyword_in_ir(path, kernel_allgather, keyword_lccl)
    assert check_keyword_in_ir(path, kernel_allgather, keyword_hccl)
    assert check_keyword_in_ir(path, kernel_allgathermatmul, keyword_lccl)
    assert not check_keyword_in_ir(path, kernel_allgathermatmul, keyword_hccl)
    print(f'===== end graph_build_2 +++', flush=True)

    # Black list for non kernels, AllGather go lccl, AllGatherMatmul go lccl.
    os.environ['MS_DISABLE_LCCL_KERNELS_LIST'] = ","
    net3 = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    out0 = net3(x)[0].asnumpy()
    out1 = net3(x)[1].asnumpy()
    print("====================================================="
          f"\n output0:\n{out0}, \n output1:\n{out1},", flush=True)
    path = './graph_lccl_kernels_list/rank_0/graph_build_3_*'
    assert check_keyword_in_ir(path, kernel_allgather, keyword_lccl)
    assert not check_keyword_in_ir(path, kernel_allgather, keyword_hccl)
    assert check_keyword_in_ir(path, kernel_allgathermatmul, keyword_lccl)
    assert not check_keyword_in_ir(path, kernel_allgathermatmul, keyword_hccl)
    print(f'===== end graph_build_3 +++', flush=True)

    # Black list for wrong spelling 'Allgather' and correct 'AllGatherMatmul',
    # Allgather go lccl, AllGatherMatmul go aclnn(no 'collective_comm_lib' tag).
    os.environ['MS_DISABLE_LCCL_KERNELS_LIST'] = "Allgather,AllGatherMatmul"
    net4 = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    out0 = net4(x)[0].asnumpy()
    out1 = net4(x)[1].asnumpy()
    print("====================================================="
          f"\n output0:\n{out0}, \n output1:\n{out1},", flush=True)
    path = './graph_lccl_kernels_list/rank_0/graph_build_4_*'
    assert check_keyword_in_ir(path, kernel_allgather, keyword_lccl)
    assert not check_keyword_in_ir(path, kernel_allgather, keyword_hccl)
    assert not check_keyword_in_ir(path, kernel_allgathermatmul, keyword_lccl)
    assert not check_keyword_in_ir(path, kernel_allgathermatmul, keyword_hccl)
    print(f'===== end graph_build_4 +++', flush=True)

    # Black list for correct 'AllGather' and  'AllGatherMatmul', Allgather go hccl, AllGatherMatmul go aclnn.
    os.environ['MS_DISABLE_LCCL_KERNELS_LIST'] = "AllGather,AllGatherMatmul"
    net5 = AllGatherMatmulNet(seq_len, hidden_size, dp, mp)
    out0 = net5(x)[0].asnumpy()
    out1 = net5(x)[1].asnumpy()
    print("====================================================="
          f"\n output0:\n{out0}, \n output1:\n{out1},", flush=True)
    path = './graph_lccl_kernels_list/rank_0/graph_build_5_*'
    assert not check_keyword_in_ir(path, kernel_allgather, keyword_lccl)
    assert check_keyword_in_ir(path, kernel_allgather, keyword_hccl)
    assert not check_keyword_in_ir(path, kernel_allgathermatmul, keyword_lccl)
    assert not check_keyword_in_ir(path, kernel_allgathermatmul, keyword_hccl)
    print(f'===== end graph_build_5 +++', flush=True)

    os.environ['MS_DEV_SAVE_GRAPHS'] = str(0)
