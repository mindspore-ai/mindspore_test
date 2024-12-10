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

import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, shape1, shape2):
    x = np.random.randn(*shape).astype(np.float32)
    batch1 = np.random.randn(*shape1).astype(np.float32)
    batch2 = np.random.randn(*shape2).astype(np.float32)
    return x, batch1, batch2


def generate_expect_forward_output(input1, batch1, batch2, beta=1, alpha=1):
    return beta * input1 + alpha * (batch1 @ batch2)


def generate_expect_backward_output(input1, batch1, batch2, beta=1, alpha=1):
    out_grad = np.ones((batch1 @ batch2).shape)
    input_grad = beta * out_grad
    if input1.size != out_grad.size:
        input_grad = (beta * out_grad).sum(0).reshape(input1.shape)
    b1_grad = alpha * (out_grad @ batch2.transpose((1, 0)))
    b2_grad = alpha * (batch1.transpose((1, 0)) @ out_grad)
    return input_grad, b1_grad, b2_grad


def addmm_forward_func(input1, batch1, batch2, beta=1, alpha=1):
    return mint.addmm(input1, batch1, batch2, beta=beta, alpha=alpha)


def addmm_backward_func(input1, batch1, batch2, beta=1, alpha=1):
    output_grad, b1_grad, b2_grad = ms.ops.grad(
        addmm_forward_func, (0, 1, 2))(input1, batch1, batch2, beta, alpha)
    return output_grad, b1_grad, b2_grad


def addmm_forward_func_tensor(input1, batch1, batch2, beta=1, alpha=1):
    return input1.addmm(batch1, batch2, beta=beta, alpha=alpha)


def addmm_backward_func_tensor(input1, batch1, batch2, beta=1, alpha=1):
    output_grad, b1_grad, b2_grad = ms.ops.grad(
        addmm_forward_func_tensor, (0, 1, 2))(input1, batch1, batch2, beta, alpha)
    return output_grad, b1_grad, b2_grad


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addmm_tensor(mode):
    """
    Feature: Ops.
    Description: test op addmm tensor.
    Expectation: expect correct result.
    """
    input_shape1 = (5, 6)
    input_shape2 = (7, 8)
    batch1_shape = (5, 7)
    batch2_shape = (7, 6)
    batch3_shape = (6, 8)
    beta = 1
    alpha = 2.0
    input1, batch1, batch2 = generate_random_input(
        input_shape1, batch1_shape, batch2_shape)
    input2, batch3, batch4 = generate_random_input(
        input_shape2, batch2_shape, batch3_shape)
    expect_forward = generate_expect_forward_output(input1, batch1, batch2)
    expect_forward2 = generate_expect_forward_output(
        input2, batch3, batch4, beta, alpha)
    expect_grad, expect_b1_grad, expect_b2_grad = generate_expect_backward_output(
        input1, batch1, batch2)
    expect_grad2, expect_b1_grad2, expect_b2_grad2 = generate_expect_backward_output(input2, batch3,
                                                                                     batch4, beta, alpha)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = addmm_forward_func_tensor(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = addmm_forward_func_tensor(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad, b1_grad, b2_grad = addmm_backward_func_tensor(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = addmm_backward_func_tensor(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    else:
        output_forward = (jit(addmm_forward_func_tensor, jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = (jit(addmm_forward_func_tensor, jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad, b1_grad, b2_grad = (jit(addmm_backward_func_tensor, jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = (jit(addmm_backward_func_tensor, jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    np.testing.assert_allclose(
        output_forward.asnumpy(), expect_forward, 4e-2, 4e-2)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, 4e-2, 4e-2)
    np.testing.assert_allclose(b1_grad.asnumpy(), expect_b1_grad, 4e-2, 4e-2)
    np.testing.assert_allclose(b2_grad.asnumpy(), expect_b2_grad, 4e-2, 4e-2)
    np.testing.assert_allclose(
        output_forward2.asnumpy(), expect_forward2, 4e-2, 4e-2)
    np.testing.assert_allclose(
        output_grad2.asnumpy(), expect_grad2, 4e-2, 4e-2)
    np.testing.assert_allclose(b1_grad2.asnumpy(), expect_b1_grad2, 4e-2, 4e-2)
    np.testing.assert_allclose(b2_grad2.asnumpy(), expect_b2_grad2, 4e-2, 4e-2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_addmm_normal(mode):
    """
    Feature: Ops.
    Description: test op addmm.
    Expectation: expect correct result.
    """
    input_shape1 = (5, 6)
    input_shape2 = (7, 8)
    batch1_shape = (5, 7)
    batch2_shape = (7, 6)
    batch3_shape = (6, 8)
    beta = 1
    alpha = 2.0
    input1, batch1, batch2 = generate_random_input(
        input_shape1, batch1_shape, batch2_shape)
    input2, batch3, batch4 = generate_random_input(
        input_shape2, batch2_shape, batch3_shape)
    expect_forward = generate_expect_forward_output(input1, batch1, batch2)
    expect_forward2 = generate_expect_forward_output(
        input2, batch3, batch4, beta, alpha)
    expect_grad, expect_b1_grad, expect_b2_grad = generate_expect_backward_output(
        input1, batch1, batch2)
    expect_grad2, expect_b1_grad2, expect_b2_grad2 = generate_expect_backward_output(input2, batch3,
                                                                                     batch4, beta, alpha)
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output_forward = addmm_forward_func(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = addmm_forward_func(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad, b1_grad, b2_grad = addmm_backward_func(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = addmm_backward_func(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    else:
        output_forward = (jit(addmm_forward_func, jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_forward2 = (jit(addmm_forward_func, jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
        output_grad, b1_grad, b2_grad = (jit(addmm_backward_func, jit_level="O0"))(
            ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2))
        output_grad2, b1_grad2, b2_grad2 = (jit(addmm_backward_func, jit_level="O0"))(
            ms.Tensor(input2), ms.Tensor(batch3), ms.Tensor(batch4), beta, alpha)
    np.testing.assert_allclose(
        output_forward.asnumpy(), expect_forward, 4e-2, 4e-2)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, 4e-2, 4e-2)
    np.testing.assert_allclose(b1_grad.asnumpy(), expect_b1_grad, 4e-2, 4e-2)
    np.testing.assert_allclose(b2_grad.asnumpy(), expect_b2_grad, 4e-2, 4e-2)
    np.testing.assert_allclose(
        output_forward2.asnumpy(), expect_forward2, 4e-2, 4e-2)
    np.testing.assert_allclose(
        output_grad2.asnumpy(), expect_grad2, 4e-2, 4e-2)
    np.testing.assert_allclose(b1_grad2.asnumpy(), expect_b1_grad2, 4e-2, 4e-2)
    np.testing.assert_allclose(b2_grad2.asnumpy(), expect_b2_grad2, 4e-2, 4e-2)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_addmm_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    input_shape1 = (3, 4)
    batch1_shape = (3, 2)
    batch2_shape = (2, 4)
    beta = 1.0
    alpha = 0.5
    input_shape2 = (5, 5)
    batch1_shape2 = (5, 4)
    batch2_shape2 = (4, 5)
    beta2 = 1.0
    alpha2 = 2.0
    input1, batch1, batch2 = generate_random_input(
        input_shape1, batch1_shape, batch2_shape)
    input2, batch1_2, batch2_2 = generate_random_input(
        input_shape2, batch1_shape2, batch2_shape2)
    TEST_OP(addmm_forward_func, [[ms.Tensor(input1), ms.Tensor(batch1), ms.Tensor(batch2), beta, alpha],
                                 [ms.Tensor(input2), ms.Tensor(batch1_2), ms.Tensor(batch2_2), beta2, alpha2]],
            'addmm', disable_input_check=True, disable_mode=['GRAPH_MODE'])
