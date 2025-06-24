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
"""test rms_norm_quant"""
import numpy as np
import pytest

from op_checker import InternalOpEnabledChecker

# should be inited before importing mindspore
op_checker = InternalOpEnabledChecker({'MS_SUBMODULE_LOG_v': '{DEVICE:1}'}, True, "./rmsnorm_quant_log")

from mindspore.ops.operations import _infer_ops as infer_ops

import mindspore as ms
from mindspore import context, Tensor, nn, ops, Parameter


class RmsNormQuantNet1Way(nn.Cell):
    """RmsNormQuantNet1Way"""
    def __init__(self, beta, gamma, scale, offset):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.scale = scale
        self.offset = offset

    def construct(self, x):
        x, _ = ops.rms_norm(x, self.gamma)
        shape = x.shape
        x = ops.add(x, self.beta)
        x = infer_ops.QuantV2()(x, self.scale, self.offset)
        x = x.reshape(shape)
        return x, shape


class RmsNormQuantNet2Way(nn.Cell):
    """RmsNormQuantNet2Way"""
    def __init__(self, gamma, beta0, beta1, scale0, scale1, offset0, offset1):
        super().__init__()
        self.gamma = gamma
        self.beta0 = beta0
        self.beta1 = beta1
        self.scale0 = scale0
        self.scale1 = scale1
        self.offset0 = offset0
        self.offset1 = offset1

    def construct(self, x):
        x, _ = ops.rms_norm(x, self.gamma)
        shape = x.shape
        x0 = ops.add(x, self.beta0)
        x1 = ops.add(x, self.beta1)
        x0 = infer_ops.QuantV2()(x0, self.scale0, self.offset0)
        x1 = infer_ops.QuantV2()(x1, self.scale1, self.offset1)
        x0 = x0.reshape(shape)
        x1 = x1.reshape(shape)
        return x0, x1, shape


KEYWORD = "kernel opname:RmsNormQuant, kernel type:internal_kernel"


def rmsnorm_quant_net_1_way(x_shape, ms_type):
    """rmsnorm_quant_net_1_way"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    beta_num = x_shape[-1]
    new_gamma_shape = [1] * len(x_shape)
    new_gamma_shape[-1] = beta_num
    beta = Parameter(Tensor(np.ones([beta_num])).astype(ms_type))
    gamma = Parameter(Tensor(np.ones([beta_num])).astype(ms_type))
    scale = Parameter(Tensor(np.random.normal(0.1, 0.5, size=[1])).astype(ms_type))
    offset = Parameter(Tensor(np.random.randint(1, 2, size=[1])).astype(ms.int8))

    net = RmsNormQuantNet1Way(beta, gamma, scale, offset)
    x_np = np.ones(x_shape)
    x = Tensor(x_np, dtype=ms_type)
    dyn_t = Tensor(shape=[None] * len(x_shape), dtype=ms_type)
    net.set_inputs(dyn_t)
    out, _ = net(x)
    out.asnumpy()
    assert op_checker.CheckOpExistByKeyword(KEYWORD)

    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "off"})
    offset_aclnn = offset.astype(ms_type)
    scale_aclnn = 1 / scale
    net_std = RmsNormQuantNet1Way(beta, gamma, scale_aclnn, offset_aclnn)
    net_std.set_inputs(dyn_t)
    out_std, _ = net_std(x)
    out_std.asnumpy()

    assert op_checker.CheckOpNotExistByKeyword(KEYWORD)

    assert np.allclose(out.asnumpy(), out_std.asnumpy(), 0.001, 1)


def rmsnorm_quant_net_2_way(x_shape, ms_type):
    """rmsnorm_quant_net_2_way"""
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})

    beta_num = x_shape[-1]
    new_gamma_shape = [1] * len(x_shape)
    new_gamma_shape[-1] = beta_num
    beta_np = np.ones([beta_num])
    gamma_np = np.ones([beta_num])
    scale_np = np.random.normal(0.1, 0.5, size=[1])
    offset_np = np.random.randint(1, 2, size=[1])
    gamma = Parameter(Tensor(gamma_np).astype(ms_type))
    beta0 = Parameter(Tensor(beta_np).astype(ms_type))
    scale0 = Parameter(Tensor(scale_np).astype(ms_type))
    offset0 = Parameter(Tensor(offset_np).astype(ms.int8))
    beta1 = Parameter(Tensor(beta_np).astype(ms_type))
    scale1 = Parameter(Tensor(scale_np).astype(ms_type))
    offset1 = Parameter(Tensor(offset_np).astype(ms.int8))

    net = RmsNormQuantNet2Way(gamma, beta0, beta1, scale0, scale1, offset0, offset1)
    x_np = np.ones(x_shape)
    x = Tensor(x_np, dtype=ms_type)
    dyn_t = Tensor(shape=[None] * len(x_shape), dtype=ms_type)
    net.set_inputs(dyn_t)
    out0, out1, _ = net(x)
    out0.asnumpy()
    assert op_checker.CheckOpExistByKeyword(KEYWORD)

    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "off"})
    offset0_aclnn = offset0.astype(ms_type)
    offset1_aclnn = offset1.astype(ms_type)
    scale0_aclnn = 1 / scale0
    scale1_aclnn = 1 / scale1
    net_std = RmsNormQuantNet2Way(gamma, beta0, beta1, scale0_aclnn, scale1_aclnn, offset0_aclnn, offset1_aclnn)
    net_std.set_inputs(dyn_t)
    out_std0, out_std1, _ = net_std(x)
    out_std0.asnumpy()
    assert op_checker.CheckOpNotExistByKeyword(KEYWORD)

    assert np.allclose(out0.asnumpy(), out_std0.asnumpy(), 0.001, 1)
    assert np.allclose(out1.asnumpy(), out_std1.asnumpy(), 0.001, 1)


@pytest.mark.level1
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('x_shape', [[32], [1, 1, 7168], [1, 1, 1, 1, 1, 1, 1, 32]])
@pytest.mark.parametrize('dtype', [ms.float16, ms.bfloat16])
def test_rmsnorm_quant_1_way(x_shape, dtype):
    """
    Feature: test rmsnorm_quant operator in graph mode
    Description: test rmsnorm_quant operator in graph mode
    Expectation: the result is correct
    """
    rmsnorm_quant_net_1_way(x_shape, dtype)


@pytest.mark.level2
@pytest.mark.platform_ascend910b
@pytest.mark.platform_ascend310p
@pytest.mark.env_onecard
@pytest.mark.parametrize('x_shape', [[1, 1, 7168]])
@pytest.mark.parametrize('dtype', [ms.float16])
def test_rmsnorm_quant_2_way(x_shape, dtype):
    """
    Feature: test rmsnorm_quant operator in graph mode
    Description: test rmsnorm_quant operator in graph mode
    Expectation: the result is correct
    """
    rmsnorm_quant_net_2_way(x_shape, dtype)
