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

import pytest
import numpy as np
import mindspore as ms
from mindspore import mint
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(input_x, vec1, vec2, beta=1, alpha=1):
    return beta * input_x + alpha * np.outer(vec1, vec2)

def generate_expect_backward_output(input_x, vec1, vec2, beta=1, alpha=1):
    out_grad = np.ones(np.outer(vec1, vec2).shape)
    input_grad = beta * out_grad
    if input_x.size != out_grad.size:
        dim = 1 if input_x.size == out_grad.shape[0] else 0
        input_grad = (beta * out_grad).sum(dim).reshape(input_x.shape)
    vec1_grad = alpha * (out_grad @ vec2)
    vec2_grad = alpha * (out_grad.transpose(1, 0) @ vec1)
    return input_grad, vec1_grad, vec2_grad

def addr_forward_func(input_x, vec1, vec2, beta=1, alpha=1):
    return mint.addr(input_x, vec1, vec2, beta=beta, alpha=alpha)

def addr_backward_func(input_x, vec1, vec2, beta=1, alpha=1):
    input_grad, vec1_grad, vec2_grad = ms.ops.grad(addr_forward_func, (0, 1, 2))(input_x, vec1, vec2, beta, alpha)
    return input_grad, vec1_grad, vec2_grad

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_addr_normal(mode):
    """
    Feature: mint.addr
    Description: Verify the result of mint.addr on platform_ascend910b
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.context.set_context(jit_level='O0')
    input_x = generate_random_input((2, 3), np.float32)
    vec1 = generate_random_input((2,), np.float32)
    vec2 = generate_random_input((3,), np.float32)
    beta = 1.0
    alpha = 1.0

    expect_forward = generate_expect_forward_output(input_x, vec1, vec2, beta, alpha)
    output_forward = addr_forward_func(ms.Tensor(input_x), ms.Tensor(vec1), ms.Tensor(vec2), beta, alpha)
    assert np.allclose(output_forward.asnumpy(), expect_forward, rtol=1e-4)

    expect_grad, expect_vec1_grad, expect_vec2_grad = generate_expect_backward_output(input_x, vec1, vec2, beta, alpha)
    output_grad, output_vec1_grad, output_vec2_grad = addr_backward_func(ms.Tensor(input_x), ms.Tensor(vec1),
                                                                         ms.Tensor(vec2), beta, alpha)
    assert np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-4)
    assert np.allclose(output_vec1_grad.asnumpy(), expect_vec1_grad, rtol=1e-4)
    assert np.allclose(output_vec2_grad.asnumpy(), expect_vec2_grad, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_addr_normal_broadcast0(mode):
    """
    Feature: mint.addr
    Description: Verify the result of mint.addr on platform_ascend910b broadcast
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.context.set_context(jit_level='O0')
    input_x = generate_random_input((4,), np.float32)
    vec1 = generate_random_input((3,), np.float32)
    vec2 = generate_random_input((4,), np.float32)
    beta = 1.0
    alpha = 1.0

    expect_forward = generate_expect_forward_output(input_x, vec1, vec2, beta, alpha)
    output_forward = addr_forward_func(ms.Tensor(input_x), ms.Tensor(vec1), ms.Tensor(vec2), beta, alpha)
    assert np.allclose(output_forward.asnumpy(), expect_forward, rtol=1e-4)

    expect_grad, expect_vec1_grad, expect_vec2_grad = generate_expect_backward_output(input_x, vec1, vec2, beta, alpha)
    output_grad, output_vec1_grad, output_vec2_grad = addr_backward_func(ms.Tensor(input_x), ms.Tensor(vec1),
                                                                         ms.Tensor(vec2), beta, alpha)
    assert np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-4)
    assert np.allclose(output_vec1_grad.asnumpy(), expect_vec1_grad, rtol=1e-4)
    assert np.allclose(output_vec2_grad.asnumpy(), expect_vec2_grad, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_addr_normal_broadcast1(mode):
    """
    Feature: mint.addr
    Description: Verify the result of mint.addr on platform_ascend910b broadcast
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.context.set_context(jit_level='O0')
    input_x = generate_random_input((5, 1), np.float32)
    vec1 = generate_random_input((5,), np.float32)
    vec2 = generate_random_input((6,), np.float32)
    beta = 1.0
    alpha = 1.0

    expect_forward = generate_expect_forward_output(input_x, vec1, vec2, beta, alpha)
    output_forward = addr_forward_func(ms.Tensor(input_x), ms.Tensor(vec1), ms.Tensor(vec2), beta, alpha)
    assert np.allclose(output_forward.asnumpy(), expect_forward, rtol=1e-4)

    expect_grad, expect_vec1_grad, expect_vec2_grad = generate_expect_backward_output(input_x, vec1, vec2, beta, alpha)
    output_grad, output_vec1_grad, output_vec2_grad = addr_backward_func(ms.Tensor(input_x), ms.Tensor(vec1),
                                                                         ms.Tensor(vec2), beta, alpha)
    assert np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-4)
    assert np.allclose(output_vec1_grad.asnumpy(), expect_vec1_grad, rtol=1e-4)
    assert np.allclose(output_vec2_grad.asnumpy(), expect_vec2_grad, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
def test_mint_addr_dynamic():
    """
    Feature: Test addr with dynamic shape in graph mode using TEST_OP.
    Description: call mint.addr with valid input and index.
    Expectation: return the correct value.
    """
    input_x1 = generate_random_input((2, 2), np.float32)
    vec11 = generate_random_input((2,), np.float32)
    vec12 = generate_random_input((2,), np.float32)

    input_x2 = generate_random_input((4, 4), np.float32)
    vec21 = generate_random_input((4,), np.float32)
    vec22 = generate_random_input((4,), np.float32)
    beta = 1.0
    alpha = 1.0

    TEST_OP(addr_forward_func,
            [[ms.Tensor(input_x1), ms.Tensor(vec11), ms.Tensor(vec12), beta, alpha],
             [ms.Tensor(input_x2), ms.Tensor(vec21), ms.Tensor(vec22), beta, alpha]],
            "addr", disable_mode=["GRAPH_MODE"], disable_input_check=True)
