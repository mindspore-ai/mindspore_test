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
import mindspore as ms
from mindspore import ops, mint, Tensor, jit, JitConfig, context
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils


def generate_random_input(shape, dtype):
    x = np.random.randn(*shape).astype(dtype)
    y = np.random.randn(*shape).astype(dtype)
    expect = np.logical_xor(x, y)
    return x, y, expect


@test_utils.run_with_cell
def logical_xor_forward_func(x, y):
    return mint.logical_xor(x, y)


@test_utils.run_with_cell
def logical_xor_backward_func(x, y):
    return ops.grad(logical_xor_forward_func, 0)(x, y)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", ['pynative', 'KBK', 'GE'])
def test_logical_xor_forward(mode):
    """
    Feature: pyboost function.
    Description: test function logical_xor forward.
    Expectation: expect correct result.
    """
    x, y, expect = generate_random_input((2, 3, 4, 5), np.float32)
    y2 = 6
    expect2 = np.logical_xor(x, y2)
    x = Tensor(x, dtype=ms.float32)
    y = Tensor(y, dtype=ms.float32)
    y2 = Tensor(y2)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = logical_xor_forward_func(x, y)
        output2 = logical_xor_forward_func(x, y2)
    elif mode == 'KBK':
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(logical_xor_forward_func, jit_config=JitConfig(jit_level="O0")))(x, y)
        output2 = (jit(logical_xor_forward_func, jit_config=JitConfig(jit_level="O0")))(x, y2)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = logical_xor_forward_func(x, y)
        output2 = logical_xor_forward_func(x, y2)
    assert np.allclose(output.asnumpy(), expect)
    assert np.allclose(output2.asnumpy(), expect2)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_logical_xor_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    ms.set_context(mode=mode, device_target="Ascend")
    x, y, expect = generate_random_input((2, 3), np.float32)
    output = logical_xor_forward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))
    assert np.allclose(output.float().asnumpy(), expect, 0.004, 0.004)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_logical_xor_dynamic_shape():
    """
    Feature: Test logical_xor op.
    Description: Test logical_xor dynamic shape.
    Expectation: the result match with expected result.
    """
    x, y, _ = generate_random_input((3, 4, 5, 6), np.int32)
    x = Tensor(x, dtype=ms.int64)
    y = Tensor(y, dtype=ms.int64)
    x2, y2, _ = generate_random_input((3, 4), np.int64)
    x2 = Tensor(x2, dtype=ms.int64)
    y2 = Tensor(y2, dtype=ms.int64)
    TEST_OP(logical_xor_forward_func, [[x, y], [x2, y2]], 'logical_xor', disable_grad=True)
