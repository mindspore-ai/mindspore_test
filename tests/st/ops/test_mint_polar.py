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
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

import numpy as np
import pytest

import mindspore as ms
import mindspore.mint as mint
import mindspore.context as context
from mindspore import Tensor
from mindspore.common import dtype as mstype


@test_utils.run_with_cell
def polar_forward(input_abs, input_angle):
    return mint.polar(input_abs, input_angle)


@test_utils.run_with_cell
def polar_backward(input_abs, input_angle):
    return ms.grad(polar_forward, grad_position=(0, 1))(input_abs, input_angle)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_2d_float32(context_mode):
    """
    Feature: mint ops Polar.
    Description: test polar forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    if context_mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": "O0"})

    abs_np = np.random.randn(2, 3).astype(np.float32)
    angle_np = np.random.randn(2, 3).astype(np.float32)

    abs_ms, angle_ms = Tensor(abs_np, mstype.float32), Tensor(angle_np, mstype.float32)
    output = polar_forward(abs_ms, angle_ms)
    expect = abs_np * (np.cos(angle_np)) + 1j * abs_np * (np.sin(angle_np))
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_3d_float64(context_mode):
    """
    Feature: mint ops polar.
    Description: test polar forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    abs_np = np.random.randn(3, 4, 5).astype(np.float32)
    angle_np = np.random.randn(3, 4, 5).astype(np.float32)
    abs_ms, angle_ms = Tensor(abs_np, mstype.float32), Tensor(angle_np, mstype.float32)
    output = polar_forward(abs_ms, angle_ms)
    expect = abs_np * (np.cos(angle_np)) + 1j * abs_np * (np.sin(angle_np))
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend910b', 'platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_forward_dynamic_shape():
    """
    Feature: mint.arange
    Description: Verify the result of arange forward with dynamic shape
    Expectation: success
    """
    inputs1_abs = ms.Tensor(np.array([[1, 10, 2], [0, 6, 1]], np.float32))
    inputs1_angle = ms.Tensor(np.array([[1.0, 3.5, 2.2], [0, 0.1, 0.2]], np.float32))

    inputs2_abs = ms.Tensor(np.array([[[5, 0.1], [0, 5.5]], [[0.1, 0.8], [5, 6]]], np.float32))
    inputs2_angle = ms.Tensor(np.array([[[5.3, -0.1], [0.3, -2.5]], [[1.2, 5.6], [3, 5]]], np.float32))

    TEST_OP(polar_forward, [[inputs1_abs, inputs1_angle], [inputs2_abs, inputs2_angle]], 'arange', disable_mode=
            ['GRAPH_MODE'], disable_grad=True, disable_yaml_check=True)
