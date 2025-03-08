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
import numpy as np
import pytest

import mindspore as ms
from mindspore.ops import operations as P
from mindspore import Tensor, nn, context
from mindspore.common.np_dtype import bfloat16

from st_utils import custom_compare

class TransBMMTransNet(nn.Cell):
    '''TransBMMTransNet for fusion'''
    def __init__(self):
        super(TransBMMTransNet, self).__init__()
        self.trans = P.Transpose()
        self.bmm = P.BatchMatMul(transpose_a=False, transpose_b=False)

    def construct(self, x1, x2, perm1, perm2):
        transpose_in = self.trans(x1, perm1)
        bmm = self.bmm(transpose_in, x2)
        out = self.trans(bmm, perm2)
        return out

def trans_bmm_trans_net(b0, b1, m, k, n, mstype=ms.float16, is_dyn=False):
    os.environ['MS_INTERNAL_ENABLE_CUSTOM_KERNEL_LIST'] = "TransposeBatchMatmulTranspose"
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    context.set_context(save_graphs=False, save_graphs_path="./trans_bmm_trans_graph")

    if ms.float16 == mstype:
        np_type = np.float16
    elif ms.bfloat16 == mstype:
        np_type = bfloat16

    net = TransBMMTransNet()

    if b0 == 0 and b1 != 0:
        perm = (1, 0, 2)
        i0_host = np.random.normal(0.0, 0.5, size=[m, b1, k]).astype(np_type)
        i1_host = np.random.normal(0.0, 0.5, size=[b1, k, n]).astype(np_type)

        if is_dyn:
            i0_host_dyn = Tensor(shape=(None, None, None), dtype=mstype)
            i1_host_dyn = Tensor(shape=(None, None, None), dtype=mstype)
            net.set_inputs(i0_host_dyn, i1_host_dyn, perm, perm)
    elif b0 != 0 and b1 == 0:
        perm = (0, 1, 2)
        i0_host = np.random.normal(0.0, 0.5, size=[b0, m, k]).astype(np_type)
        i1_host = np.random.normal(0.0, 0.5, size=[k, n]).astype(np_type)

        if is_dyn:
            i0_host_dyn = Tensor(shape=(None, None, None), dtype=mstype)
            i1_host_dyn = Tensor(shape=(None, None), dtype=mstype)
            net.set_inputs(i0_host_dyn, i1_host_dyn, perm, perm)
    elif b0 == 0 and b1 == 0:
        perm = (0, 1)
        i0_host = np.random.normal(0.0, 0.5, size=[m, k]).astype(np_type)
        i1_host = np.random.normal(0.0, 0.5, size=[k, n]).astype(np_type)

        if is_dyn:
            i0_host_dyn = Tensor(shape=(None, None), dtype=mstype)
            i1_host_dyn = Tensor(shape=(None, None), dtype=mstype)
            net.set_inputs(i0_host_dyn, i1_host_dyn, perm, perm)
    else:
        perm = (0, 2, 1, 3)
        i0_host = np.random.normal(0.0, 0.5, size=[b0, m, b1, k]).astype(np_type)
        i1_host = np.random.normal(0.0, 0.5, size=[b1, k, n]).astype(np_type)

        if is_dyn:
            i0_host_dyn = Tensor(shape=(None, None, None, None), dtype=mstype)
            i1_host_dyn = Tensor(shape=(None, None, None), dtype=mstype)
            net.set_inputs(i0_host_dyn, i1_host_dyn, perm, perm)

    i0_host_fp32 = i0_host.astype(np.float32)
    i1_host_fp32 = i1_host.astype(np.float32)
    trans_out = i0_host_fp32.transpose(perm)
    bmm = np.matmul(trans_out, i1_host_fp32)
    expect = bmm.transpose(perm)

    input1 = ms.Tensor(i0_host, mstype)
    input2 = ms.Tensor(i1_host, mstype)
    output = net(input1, input2, perm, perm)

    output_fp32 = output.astype(ms.float32)
    output_np = output_fp32.asnumpy()
    res = custom_compare(expect, output_np, mstype)
    assert res, "TransposeBatchMatmulTranspose compare fail."

@pytest.mark.level1
@pytest.mark.parametrize('b0', [1, 2])
@pytest.mark.parametrize('b1', [1, 4])
@pytest.mark.parametrize('mstype', [ms.float16, ms.bfloat16])
@pytest.mark.platform_ascend910b
def test_transpose_batch_matmul_transpose_with_b0_b1(b0, b1, mstype):
    """
    Feature: test transpose operator in graph mode
    Description: test transpose.
    Expectation: the result is correct
    """
    trans_bmm_trans_net(b0, b1, 64, 128, 256, mstype)
    trans_bmm_trans_net(b0, b1, 510, 510, 510, mstype)

@pytest.mark.level1
@pytest.mark.parametrize('b1', [2, 3])
@pytest.mark.parametrize('mstype', [ms.float16, ms.bfloat16])
@pytest.mark.platform_ascend910b
def test_transpose_batch_matmul_transpose_with_b1(b1, mstype):
    """
    Feature: test transpose operator in graph mode
    Description: test transpose.
    Expectation: the result is correct
    """
    trans_bmm_trans_net(0, b1, 64, 128, 256, mstype)
    trans_bmm_trans_net(0, b1, 70, 70, 70, mstype)
