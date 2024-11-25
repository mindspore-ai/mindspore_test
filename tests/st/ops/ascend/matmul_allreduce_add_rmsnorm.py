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

import numpy as np
import pytest
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops.operations import _infer_ops as infer_ops
from mindspore import nn, Tensor, context, JitConfig, mutable
import mindspore.communication.management as D

def generate_expect_output(x1, x2, residual, gamma, epsilon, bias=0):
    mm_out = np.dot(x1, x2) + bias
    allreduce_out = 2 * mm_out
    y = allreduce_out + residual
    d = y.size
    rms_y = np.sqrt(np.sum(y ** 2) / d + epsilon)
    norm_out = (y / rms_y) * gamma
    return y, norm_out

class Matmul_AllReduce_Add_RmsNormNet(nn.Cell):
    def __init__(self, epsilon):
        super().__init__()
        self.matmul = P.MatMul(transpose_a=False, transpose_b=True)
        self.allreduce = P.AllReduce()
        self.reshape = P.Reshape()
        self.rmsnorm = P.RmsNorm(epsilon)

    def construct(self, x1, x2, residual, gamma, xshape, yshape, zshape):
        mm_out = self.matmul(x1, x2)
        allreduce_out = self.allreduce(mm_out)
        out_shape = (mutable(xshape),) + (mutable(yshape),) + (mutable(zshape),)
        reshape_out = self.reshape(allreduce_out, out_shape)
        y = residual + reshape_out
        rmsnormout = self.rmsnorm(y, gamma)
        return y, rmsnormout[0]

class MatmulAllReduceAddRmsNormNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul_allreduce_addrmsnorm = infer_ops.MatmulAllReduceAddRmsNorm()

    def construct(self, x1, x2, bias, residual, gamma, epsilon, group,
                  reduceOp='sum', commTurn=0, streamMode=1):
        y, normOut = self.matmul_allreduce_addrmsnorm(x1, x2, bias, residual, gamma, epsilon, group,
                                                      reduceOp, commTurn, streamMode)
        return y, normOut

@pytest.mark.parametrize('tensor_type', [mstype.float16, mstype.bfloat16])
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_matmul_allreduce_addrmsnorm_forward(mode, tensor_type):
    """
    Feature: Test MatmulAllReduceAddRmsNorm forward.
    Description: Test in kbk and pynative mode with dtype float16 and bfloat16
    Expectation: Run success
    """
    context.set_context(device_target="Ascend", mode=mode)
    D.init()

    x1 = np.ones((2, 2))
    x2 = np.ones((2, 2))
    bias = np.array([1., 1.], np.float32)
    residual = np.ones((1, 2, 2)) * 2
    gamma = np.array([2., 2.], np.float32)

    x1_tensor = Tensor(x1, dtype=tensor_type)
    x2_tensor = Tensor(x2, dtype=tensor_type)
    bias_tensor = Tensor(bias, dtype=tensor_type)
    residual_tensor = Tensor(residual, dtype=tensor_type)
    gamma_tensor = Tensor(gamma, dtype=tensor_type)
    group = D.HCCL_WORLD_COMM_GROUP
    epsilon = 1e-6

    # calculate
    matmulallreduceaddrmsnormnet = MatmulAllReduceAddRmsNormNet()
    if mode == context.GRAPH_MODE:
        matmulallreduceaddrmsnormnet.set_jit_config(JitConfig(jit_level="O0", infer_boost="on"))
    expect_output = generate_expect_output(x1, x2, residual, gamma, epsilon, bias)
    output = matmulallreduceaddrmsnormnet(x1_tensor, x2_tensor, bias_tensor,
                                          residual_tensor, gamma_tensor, epsilon, group)

    # compare
    assert np.allclose(output[0].float().asnumpy(), expect_output[0], rtol=5e-3, atol=5e-3)
    assert np.allclose(output[1].float().asnumpy(), expect_output[1], rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize('tensor_type', [mstype.float16, mstype.bfloat16])
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE, context.GRAPH_MODE])
def test_matmul_allreduce_addrmsnorm_forward_dynamic_shape(mode, tensor_type):
    """
    Feature: Test MatmulAllReduceAddRmsNorm forward with dynamic shape input
    Description: Test in kbk and pynative mode with dtype float16 and bfloat16
    Expectation: Run success
    """
    context.set_context(device_target="Ascend", mode=mode)
    D.init()

    x1 = np.ones((2, 2))
    x2 = np.ones((2, 2))
    bias = np.array([1., 1.], np.float32)
    residual = np.ones((1, 2, 2)) * 2
    gamma = np.array([2., 2.], np.float32)

    x1_tensor = Tensor(x1, dtype=tensor_type)
    x2_tensor = Tensor(x2, dtype=tensor_type)
    bias_tensor = Tensor(bias, dtype=tensor_type)
    residual_tensor = Tensor(residual, dtype=tensor_type)
    gamma_tensor = Tensor(gamma, dtype=tensor_type)
    group = D.HCCL_WORLD_COMM_GROUP
    epsilon = 1e-6

    x1_dyn_shape = [None, None]
    x2_dyn_shape = [None, None]
    bias_dyn_shape = [None]
    residual_dyn_shape = [None, None, None]
    gamma_dyn_shape = [None]
    x1_dyn = Tensor(shape=x1_dyn_shape, dtype=tensor_type)
    x2_dyn = Tensor(shape=x2_dyn_shape, dtype=tensor_type)
    bias_dyn = Tensor(shape=bias_dyn_shape, dtype=tensor_type)
    residual_dyn = Tensor(shape=residual_dyn_shape, dtype=tensor_type)
    gamma_dyn = Tensor(shape=gamma_dyn_shape, dtype=tensor_type)

    # calculate
    matmulallreduceaddrmsnormnet = MatmulAllReduceAddRmsNormNet()
    if mode == context.GRAPH_MODE:
        matmulallreduceaddrmsnormnet.set_jit_config(JitConfig(jit_level="O0", infer_boost="on"))
    expect_output = generate_expect_output(x1, x2, residual, gamma, epsilon, bias)
    matmulallreduceaddrmsnormnet.set_inputs(x1_dyn, x2_dyn, bias_dyn, residual_dyn, gamma_dyn, epsilon, group)
    output = matmulallreduceaddrmsnormnet(x1_tensor, x2_tensor, bias_tensor,
                                          residual_tensor, gamma_tensor, epsilon, group)

    # compare
    assert np.allclose(output[0].float().asnumpy(), expect_output[0], rtol=5e-3, atol=5e-3)
    assert np.allclose(output[1].float().asnumpy(), expect_output[1], rtol=5e-3, atol=5e-3)


@pytest.mark.parametrize('tensor_type', [mstype.float16])
@pytest.mark.parametrize('mode', [context.GRAPH_MODE])
def test_matmul_allreduce_addrmsnorm_forward_fusion(mode, tensor_type):
    """
    Feature: Test MatmulAllReduceAddRmsNorm forward ir fusion pass
    Description: Test in kbk mode with dtype float16 and bfloat16
    Expectation: Run success
    """
    context.set_context(device_target="Ascend", mode=mode, save_graphs=True)
    D.init()

    x1 = np.ones((2, 2))
    x2 = np.ones((2, 2))
    residual = np.ones((1, 2, 2)) * 2
    gamma = np.array([2., 2.], np.float32)

    x1_tensor = Tensor(x1, dtype=tensor_type)
    x2_tensor = Tensor(x2, dtype=tensor_type)
    residual_tensor = Tensor(residual, dtype=tensor_type)
    gamma_tensor = Tensor(gamma, dtype=tensor_type)
    epsilon = 1e-6

    # calculate
    matmul_allreduce_add_rmsnormnet = Matmul_AllReduce_Add_RmsNormNet(epsilon)
    matmul_allreduce_add_rmsnormnet.set_jit_config(JitConfig(jit_level="O0", infer_boost="on"))
    expect_output = generate_expect_output(x1, x2, residual, gamma, epsilon)
    output = matmul_allreduce_add_rmsnormnet(x1_tensor, x2_tensor,
                                             residual_tensor, gamma_tensor, 1, 2, 2)

    # compare
    assert np.allclose(output[0].float().asnumpy(), expect_output[0], rtol=5e-3, atol=5e-3)
    assert np.allclose(output[1].float().asnumpy(), expect_output[1], rtol=5e-3, atol=5e-3)
