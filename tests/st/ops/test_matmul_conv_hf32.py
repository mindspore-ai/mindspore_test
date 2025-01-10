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
from mindspore import Tensor, context
from mindspore import ops, mint
from tests.device_utils import set_device
from tests.st.utils import test_utils


@test_utils.run_with_cell
def matmul_forward_func(x, y):
    return mint.matmul(x, y)


@test_utils.run_with_cell
def bmm_forward_func(x, y):
    return mint.bmm(x, y)


@test_utils.run_with_cell
def baddbmm_forward_func(input_x, batch1, batch2):
    return mint.baddbmm(input_x, batch1, batch2)


@test_utils.run_with_cell
def addmm_forward_func(input_x, mat1, mat2):
    return ops.auto_generate.addmm_op(input_x, mat1, mat2, 1, 1)


@test_utils.run_with_cell
def mv_forward_func(m, v):
    return ops.auto_generate.mv(m, v)


@test_utils.run_with_cell
def conv2d_forward_func(input_x, weight, bias):
    return mint.conv2d(input_x, weight, bias)


def set_mode(mode):
    """
    set mode
    """
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@pytest.mark.parametrize("mode", ["KBK", "PYBOOST"])
def test_hf32(mode):
    """
    Feature: hf32
    Description: test hf32 with aclnn api
    Expectation: expect correct result.
    """
    set_mode(mode)
    set_device()
    ms.device_context.ascend.op_precision.matmul_allow_hf32(True)
    ms.device_context.ascend.op_precision.conv_allow_hf32(True)

    # mint.matmul
    x = np.array([[1.7640524, 0.4001572, 0.978738],
                  [2.2408931, 1.867558, -0.9772779],
                  [0.95008844, -0.1513572, -0.10321885]]).astype(np.float32)
    x = Tensor(x)
    y = np.array([[0.41059852, 0.14404356, 1.4542735],
                  [0.7610377, 0.12167501, 0.44386324],
                  [0.33367434, 1.4940791, -0.20515826]]).astype(np.float32)
    y = Tensor(y)
    expect = np.array([[1.3554808, 1.7652068, 2.5420902],
                       [2.0155735, -0.91013855, 4.2883935],
                       [0.24057126, -0.03575936, 1.335669]]).astype(np.float32)
    output = matmul_forward_func(x, y)
    assert np.allclose(output.asnumpy(), expect, 1e-05, 1e-05)

    # mint.bmm
    mat1 = x.unsqueeze(0)
    mat2 = y.unsqueeze(0)
    expect = np.expand_dims(expect, 0)
    output = bmm_forward_func(mat1, mat2)
    assert np.allclose(output.asnumpy(), expect, 1e-05, 1e-05)

    # ops.auto_generate.addmm
    input_x = np.array([[0.3130677, -0.85409576, -2.5529897]]).astype(np.float32)
    input_x = Tensor(input_x)
    mat1 = x
    mat2 = y
    expect = np.array([[1.6685485, 0.9111111, -0.01089956],
                       [2.3286412, -1.7642343, 1.7354035],
                       [0.553639, -0.88985515, -1.2173207]]).astype(np.float32)
    output = addmm_forward_func(input_x, mat1, mat2)
    assert np.allclose(output.asnumpy(), expect, 1e-05, 1e-05)

    # ops.auto_generate.inplace_addmm
    input_x = np.array([[1.7640524, 0.4001572, 0.978738],
                        [0.3130677, -0.85409576, -2.5529897],
                        [0.41059852, 0.14404356, 1.4542735]]).astype(np.float32)
    input_x = Tensor(input_x)
    expect = np.array([[3.119533, 2.165364, 3.5208282],
                       [2.3286412, -1.7642343, 1.7354038],
                       [0.6511698, 0.10828421, 2.7899425]]).astype(np.float32)
    input_x.addmm_(mat1, mat2)
    assert np.allclose(input_x.asnumpy(), expect, 1e-05, 1e-05)

    # ops.auto_generate.mv
    m = x
    v = np.array([0.41059852, 0.14404356, 1.4542735]).astype(np.float32)
    v = Tensor(v)
    expect = np.array([2.205297, -0.2317195, 0.21831065]).astype(np.float32)
    output = mv_forward_func(m, v)
    assert np.allclose(output.asnumpy(), expect, 1e-05, 1e-05)

    # mint.baddbmm
    input_x = np.array([0.3130677, -0.85409576, -2.5529897]).astype(np.float32)
    input_x = np.reshape(input_x, (1, 1, 3))
    input_x = Tensor(input_x)
    batch1 = x.unsqueeze(0)
    batch2 = y.unsqueeze(0)
    expect = np.array([[[1.6685485, 0.9111111, -0.01089956],
                        [2.3286412, -1.7642343, 1.7354035],
                        [0.553639, -0.88985515, -1.2173207]]]).astype(np.float32)
    output = baddbmm_forward_func(input_x, batch1, batch2)
    assert np.allclose(output.asnumpy(), expect, 1e-05, 1e-05)

    # mint.nn.functional.conv2d
    input_x = np.arange(54).astype(np.float32).reshape(3, 2, 3, 3)
    input_x = Tensor(input_x)
    weight = np.array([[-1.090221803810641, -0.044567894776783905],
                       [0.04005113957734308, 0.22892450020231897]]).astype(np.float32)
    weight = np.reshape(weight, (2, 2, 1, 1))
    weight = Tensor(weight)
    bias = np.array([0.729725082405579, 0.6472988621466479]).astype(np.float32)
    bias = Tensor(bias)
    expect = np.array([[0.32858676, -0.8063162, -1.9412191],
                       [-3.076122, -4.211025, -5.345928],
                       [-6.480831, -7.615734, -8.750637],
                       [2.7077847, 2.9767818, 3.245779],
                       [3.5147762, 3.7837734, 4.0527706],
                       [4.321768, 4.590765, 4.859762],
                       [-20.099667, -21.23457, -22.369473],
                       [-23.504375, -24.639278, -25.774181],
                       [-26.909084, -28.043987, -29.17889],
                       [7.549734, 7.8187313, 8.0877285],
                       [8.356726, 8.625723, 8.89472],
                       [9.163717, 9.432714, 9.701712],
                       [-40.52792, -41.662823, -42.797726],
                       [-43.93263, -45.06753, -46.202435],
                       [-47.337337, -48.47224, -49.607143],
                       [12.391684, 12.660681, 12.929678],
                       [13.198675, 13.467672, 13.73667],
                       [14.005667, 14.274664, 14.543661]]).astype(np.float32)
    expect = np.reshape(expect, (3, 2, 3, 3))
    output = conv2d_forward_func(input_x, weight, bias)
    assert np.allclose(output.asnumpy(), expect, 1e-05, 1e-05)
