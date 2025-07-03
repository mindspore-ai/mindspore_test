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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore import mint, jit
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_ones_input(shape, dtype):
    return np.ones(shape, dtype=dtype)


@test_utils.run_with_cell
def dropout2d_forward_func(x, p, training):
    return mint.nn.functional.dropout2d(x, p, training)


@test_utils.run_with_cell
def dropout2d_backward_func(x, p, training):
    return ms.grad(dropout2d_forward_func, (0,))(x, p, training)


@jit(backend="ms_backend")
@test_utils.run_with_cell
def nn_dropout2d_backward_func(net, x):
    return ms.grad(net, (0,))(x)


def compare_output(x, p, output):
    keep_prob = 1 - p
    if output.dtype == ms.bfloat16:
        output_np = output.float().asnumpy()
    else:
        output_np = output.asnumpy()
    elem_count = x.size
    keep_count = np.count_nonzero(output_np)
    assert (elem_count * (keep_prob - 0.02)) < keep_count < (elem_count * (keep_prob + 0.02))

    expect_sum = np.array(keep_count / (1 - p), dtype=np.float64)
    output_sum = np.sum(output_np.astype(np.float64))

    if output.dtype == ms.bfloat16:
        np.testing.assert_allclose(output_sum, expect_sum, rtol=1e-2)
    else:
        np.testing.assert_allclose(output_sum, expect_sum, rtol=1e-3)


def compare_grad(x, p, grad):
    keep_prob = 1 - p
    if grad.dtype == ms.bfloat16:
        grad_np = grad.float().asnumpy()
    else:
        grad_np = grad.asnumpy()
    elem_count = x.size
    keep_count = np.count_nonzero(grad_np)
    assert (elem_count * (keep_prob - 0.02)) < keep_count < (elem_count * (keep_prob + 0.02))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape', [(100, 100, 100, 100)])
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_func_dropout2d(shape, mode):
    """
    Feature: standard forward, backward features.
    Description: test function dropout2d.
    Expectation: expect correct result.
    """
    x = generate_ones_input(shape, np.float32)
    p = 0.3
    training = True
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    output = dropout2d_forward_func(ms.Tensor(x), p, training)
    output_grad = dropout2d_backward_func(ms.Tensor(x), p, training)

    assert output.shape == shape
    assert output.dtype == ms.float32
    assert output_grad.shape == shape
    assert output_grad.dtype == ms.float32
    compare_output(x, p, output)
    compare_grad(x, p, output_grad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('shape', [(100, 100, 100, 100)])
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_nn_Dropout2d(shape, mode):
    """
    Feature: standard forward, backward features.
    Description: test function Dropout2d.
    Expectation: expect correct result.
    """
    x = generate_ones_input(shape, np.float32)
    p = 0.3
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    net = mint.nn.Dropout2d(p)
    net.set_train()
    output = net(ms.Tensor(x))
    output_grad = nn_dropout2d_backward_func(net, ms.Tensor(x))

    assert output.shape == shape
    assert output.dtype == ms.float32
    assert output_grad.shape == shape
    assert output_grad.dtype == ms.float32
    compare_output(x, p, output)
    compare_grad(x, p, output_grad)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('shape', [(100, 100, 100, 100)])
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_func_dropout2d_bfloat16(shape, mode):
    """
    Feature: test dropout2d functional API.
    Description: testcase for dropout2d functional API.
    Expectation: the result match with expected result.
    """
    x = generate_ones_input(shape, np.float32)
    p = 0.3
    training = True
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    output = dropout2d_forward_func(ms.Tensor(x, dtype=ms.bfloat16), p, training)
    output_grad = dropout2d_backward_func(ms.Tensor(x, dtype=ms.bfloat16), p, training)

    assert output.shape == shape
    assert output.dtype == ms.bfloat16
    assert output_grad.shape == shape
    assert output_grad.dtype == ms.bfloat16
    compare_output(x, p, output)
    compare_grad(x, p, output_grad)
