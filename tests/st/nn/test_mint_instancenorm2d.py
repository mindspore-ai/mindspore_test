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
import mindspore.mint.nn as nn
from mindspore import Tensor, context
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


@test_utils.run_with_cell
def forward_instance_norm_2d_net(input_x, num_features, eps=1e-5, momentum=0.1, affine=True,
                                 track_running_stats=True, dtype=None):
    net = nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                            track_running_stats=track_running_stats, dtype=dtype)
    return net(input_x)


@test_utils.run_with_cell
def forward_instance_norm_2d_for_dyn(input_x):
    net = nn.InstanceNorm2d(4)
    return net(input_x)


@test_utils.run_with_cell
def grad_instance_norm_2d_net(input_x, num_features, eps=1e-5, momentum=0.1, affine=True,
                              track_running_stats=True, dtype=None):
    net = nn.InstanceNorm2d(num_features, eps=eps, momentum=momentum, affine=affine,
                            track_running_stats=track_running_stats, dtype=dtype)
    return ms.grad(net)(input_x)


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
def test_instance_norm_2d(mode):
    """
    Feature: InstanceNorm2d
    Description: Verify the result of InstanceNorm2d.
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
    num_features = 2
    x = Tensor(input_x, dtype=ms.float32)
    output = forward_instance_norm_2d_net(x, num_features)
    out_grad = grad_instance_norm_2d_net(x, num_features)

    assert np.allclose(expect_output_shape, output.shape)
    np.testing.assert_allclose(
        output.asnumpy(), expect_output, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(
        out_grad.asnumpy(), expect_output_grad, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_instancenorm2d_dyn():
    """
    Feature: Dynamic shape of InstanceNorm1d
    Description: test InstanceNorm1d with dynamic rank/shape.
    Expectation: success
    """
    in1 = Tensor(np.random.randn(4, 4, 3, 2), dtype=ms.float32)
    in2 = Tensor(np.random.randn(4, 4, 2), dtype=ms.float32)
    TEST_OP(forward_instance_norm_2d_for_dyn, [[in1], [in2]], '', disable_input_check=True,
            disable_yaml_check=True, disable_mode=['GRAPH_MODE', 'GRAPH_MODE_O0'])
