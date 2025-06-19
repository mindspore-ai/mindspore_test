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

"""test gather_pre_rms_norm"""

import numpy as np
import pytest

import mindspore as ms
from mindspore import nn, context
from mindspore import ops
from tests.mark_utils import arg_mark

HALF_FLOAT_MIN = -5.0
HALF_FLOAT_MAX = 5.0


class GatherPreRmsNormNet(nn.Cell):
    """GatherPreRmsNormNet"""
    def construct(self, x, res_in, indices, gamma, eps):
        """construct"""
        res_in_gather = ops.gather(res_in, indices, axis=0, batch_dims=0)
        res_out = x + res_in_gather
        y, _ = ops.rms_norm(res_out, gamma, epsilon=eps)
        return y, res_out


def golden_np(dim_m, dim_n, dim_res, epsilon=1e-5):
    np.random.seed(1222)

    x = np.random.uniform(low=HALF_FLOAT_MIN, high=HALF_FLOAT_MAX, size=(dim_m, dim_n))
    res_in = np.random.uniform(low=HALF_FLOAT_MIN, high=HALF_FLOAT_MAX, size=(dim_res, dim_n))
    indices = np.random.randint(0, dim_res, size=dim_m)
    gamma = np.random.uniform(low=HALF_FLOAT_MIN, high=HALF_FLOAT_MAX, size=dim_n)

    res_in_after = res_in[indices]
    res_out = x + res_in_after

    square_mean = np.mean(res_out ** 2, axis=-1, keepdims=True)
    rms = np.sqrt(square_mean + epsilon)

    y = res_out / rms * gamma

    inputs = [x, res_in, indices, gamma]
    outputs = [y, res_out]
    return inputs, outputs


def run_test(dim_m, dim_n, dim_res, epsilon, dtype, is_dyn=False, mode=context.GRAPH_MODE):
    context.set_context(mode=mode, device_target="Ascend")
    context.set_context(jit_config={"jit_level": "O0", "infer_boost": "on"})
    context.set_context(save_graphs=1, save_graphs_path='./gather_pre_rms_norm')

    inputs, outputs = golden_np(dim_m, dim_n, dim_res, epsilon)

    x = ms.Tensor(inputs[0], dtype)
    res_in = ms.Tensor(inputs[1], dtype)
    indices = ms.Tensor(inputs[2], ms.int32)
    gamma = ms.Tensor(inputs[3], dtype)

    net = GatherPreRmsNormNet()

    if is_dyn:
        x_dyn = ms.Tensor(shape=[None, None], dtype=dtype)
        res_in_dyn = ms.Tensor(shape=[None, None], dtype=dtype)
        indices_dyn = ms.Tensor(shape=[None], dtype=ms.int32)
        gamma_dyn = ms.Tensor(shape=[None], dtype=dtype)
        net.set_inputs(x=x_dyn, res_in=res_in_dyn, indices=indices_dyn, gamma=gamma_dyn)

    y, res_out = net(x, res_in, indices, gamma, epsilon)

    atol = 0.001 if dtype == ms.float16 else 0.005

    y_allclose = np.allclose(y.astype(ms.float32).asnumpy(), outputs[0], atol=atol)
    res_out_allclose = np.allclose(res_out.astype(ms.float32).asnumpy(), outputs[1], atol=atol)

    if y_allclose and res_out_allclose:
        return True

    print(f'np y: {outputs[0]}')
    print(f'ms y: {y.astype(ms.float32).asnumpy()}')

    print(f'np res_out: {outputs[1]}')
    print(f'ms res_out: {res_out.astype(ms.float32).asnumpy()}')
    return False


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('is_dyn', [True, False])
def test_gatherprermsnorm_f16_small_indices_case0(is_dyn):
    """
    Feature: test gather_pre_rms_norm fusion in graph mode
    Description: test gather_pre_rms_norm
    Expectation: the result is correct
    """
    res_tokens = 3
    ind_tokens = 4
    hidden_size = 16
    epsilon = 1e-5
    run_test(ind_tokens, hidden_size, res_tokens, epsilon, ms.float16, is_dyn=is_dyn)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('is_dyn', [True, False])
def test_gatherprermsnorm_f16_small_indices_case1(is_dyn):
    """
    Feature: test gather_pre_rms_norm fusion in graph mode
    Description: test gather_pre_rms_norm
    Expectation: the result is correct
    """
    res_tokens = 97
    ind_tokens = 101
    hidden_size = 7680
    epsilon = 1e-5
    run_test(ind_tokens, hidden_size, res_tokens, epsilon, ms.float16, is_dyn=is_dyn)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('is_dyn', [True, False])
def test_gatherprermsnorm_f16_small_indices_case2(is_dyn):
    """
    Feature: test gather_pre_rms_norm fusion in graph mode
    Description: test gather_pre_rms_norm
    Expectation: the result is correct
    """
    res_tokens = 300
    ind_tokens = 533
    hidden_size = 7168
    epsilon = 1e-8
    run_test(ind_tokens, hidden_size, res_tokens, epsilon, ms.float16, is_dyn=is_dyn)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('is_dyn', [True, False])
def test_gatherprermsnorm_f16_large_indices_case(is_dyn):
    """
    Feature: test gather_pre_rms_norm fusion in graph mode
    Description: test gather_pre_rms_norm
    Expectation: the result is correct
    """
    res_tokens = 99999
    ind_tokens = 240000
    hidden_size = 2560
    epsilon = 1e-8
    run_test(ind_tokens, hidden_size, res_tokens, epsilon, ms.float16, is_dyn=is_dyn)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('is_dyn', [True, False])
def test_gatherprermsnorm_bf16_small_indices_case0(is_dyn):
    """
    Feature: test gather_pre_rms_norm fusion in graph mode
    Description: test gather_pre_rms_norm
    Expectation: the result is correct
    """
    res_tokens = 16
    ind_tokens = 16
    hidden_size = 7168
    epsilon = 1e-5
    run_test(ind_tokens, hidden_size, res_tokens, epsilon, ms.bfloat16, is_dyn=is_dyn)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('is_dyn', [True, False])
def test_gatherprermsnorm_bf16_small_indices_case1(is_dyn):
    """
    Feature: test gather_pre_rms_norm fusion in graph mode
    Description: test gather_pre_rms_norm
    Expectation: the result is correct
    """
    res_tokens = 233
    ind_tokens = 334
    hidden_size = 5120
    epsilon = 1e-8
    run_test(ind_tokens, hidden_size, res_tokens, epsilon, ms.bfloat16, is_dyn=is_dyn)
