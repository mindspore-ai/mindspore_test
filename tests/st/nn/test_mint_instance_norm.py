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

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, context, mint
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


@test_utils.run_with_cell
def forward_instance_norm_net(input_x, running_mean, running_var, weight=None, bias=None):
    running_mean = running_mean * 1
    running_var = running_var * 1
    return mint.nn.functional.instance_norm(input_x, running_mean, running_var, weight=weight, bias=bias)


@test_utils.run_with_cell
def grad_instance_norm_net(input_x, running_mean, running_var, weight=None, bias=None):
    return ms.grad(forward_instance_norm_net)(input_x, running_mean, running_var, weight=weight, bias=bias)


def set_mode(mode):
    """
    set mode
    """
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative'])
def test_instance_norm(mode):
    """
    Feature: instance_norm
    Description: Verify the result of instance_norm.
    Expectation: success
    """

    set_mode(mode)

    input_x = np.arange(0, 2 * 2 * 2 * 2, 1).reshape(2, 2, 2, 2).astype(np.float32)
    expect_output = np.array([[[[-1.3416, -0.4472], [0.4472, 1.3416]],
                               [[-1.3416, -0.4472], [0.4472, 1.3416]]],
                              [[[-1.3416, -0.4472], [0.4472, 1.3416]],
                               [[-1.3416, -0.4472], [0.4472, 1.3416]]]]).astype(np.float32)
    expect_output_grad = np.array([[[[0., 0.], [0., 0.]],
                                    [[0., 0.], [0., 0.]]],
                                   [[[0., 0.], [0., 0.]],
                                    [[0., 0.], [0., 0.]]]]).astype(np.float32)
    expect_output_shape = (2, 2, 2, 2)
    x = Tensor(input_x, dtype=ms.float32)
    running_mean = Tensor([0, 0], dtype=ms.float32)
    running_var = Tensor([2, 2], dtype=ms.float32)
    output = forward_instance_norm_net(x, running_mean, running_var)
    out_grad = grad_instance_norm_net(x, running_mean, running_var)

    assert np.allclose(expect_output_shape, output.shape)
    np.testing.assert_allclose(
        output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        out_grad.asnumpy(), expect_output_grad, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_instancenorm_dyn():
    """
    Feature: Dynamic shape of InstanceNorm
    Description: test InstanceNorm with dynamic rank/shape.
    Expectation: success
    """
    in1 = Tensor(np.random.randn(4, 2, 3, 2), dtype=ms.float32)
    in2 = Tensor(np.random.randn(4, 2, 2), dtype=ms.float32)

    running_mean1 = Tensor([0, 0], dtype=ms.float32)
    running_var1 = Tensor([2, 2], dtype=ms.float32)
    running_mean2 = Tensor([1, 1], dtype=ms.float32)
    running_var2 = Tensor([3, 3], dtype=ms.float32)
    weight1 = Tensor([0, 0], dtype=ms.float32)
    weight2 = Tensor([2, 2], dtype=ms.float32)
    bias1 = Tensor([1, 5], dtype=ms.float32)
    bias2 = Tensor([4, 2], dtype=ms.float32)

    TEST_OP(grad_instance_norm_net, [[in1, running_mean1, running_var1, weight1, bias1],
                                     [in2, running_mean2, running_var2, weight2, bias2]],
            '', disable_input_check=True,
            disable_yaml_check=True,
            disable_mode=['GRAPH_MODE', 'GRAPH_MODE_O0'])
