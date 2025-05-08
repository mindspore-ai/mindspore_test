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
from mindspore.common import dtype as mstype
from mindspore import ops, mint, Tensor, jit, context
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils


@test_utils.run_with_cell
def full_like_forward_func(x, fill_value, dtype=None):
    y = mint.full_like(x, fill_value, dtype=dtype)
    return y

@test_utils.run_with_cell
def full_like_backward_func(x, fill_value, dtype=None):
    grad = ops.grad(full_like_forward_func, (0,))(x, fill_value, dtype=dtype)
    return grad


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", ['pynative', 'KBK'])
def test_full_like_normal(mode):
    """
    Feature: mint.full_like
    Description: Verify the result of mint.full_like
    Expectation: success
    """
    x = Tensor([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]], dtype=mstype.int32)
    fill_value = 11
    dtype = mstype.float32
    expect_out = np.array([[[11, 11, 11, 11], [11, 11, 11, 11], [11, 11, 11, 11]]], dtype=np.float32)
    expect_grad = 0

    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = full_like_forward_func(x, fill_value, dtype)
        grad = full_like_backward_func(x, fill_value, dtype)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(full_like_forward_func, backend="ms_backend", jit_level="O0"))(x, fill_value, dtype)
        grad = (jit(full_like_backward_func, backend="ms_backend", jit_level="O0"))(x, fill_value, dtype)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-3)
    np.testing.assert_allclose(grad.asnumpy(), expect_grad, rtol=1e-3)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
def test_full_like_dynamic_shape():
    """
    Feature: Test full_like with dynamic shape in graph mode.
    Description: call mint.full_like with valid input and index.
    Expectation: return the correct value.
    """
    tensor1 = Tensor(np.arange(6).reshape(2, 3), dtype=mstype.float32)
    tensor2 = Tensor(np.arange(24).reshape(2, 3, 4), dtype=mstype.float32)
    fill_value1 = 2
    fill_value2 = 3

    TEST_OP(full_like_forward_func, [[tensor1, fill_value1], [tensor2, fill_value2]], disable_mode=['GRAPH_MODE_GE'])
