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
from mindspore import ops, Tensor
from mindspore.ops import sign
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.sign(x)


@test_utils.run_with_cell
def sign_forward_func(x):
    return sign(x)


@test_utils.run_with_cell
def sign_vmap_func(x):
    return ops.vmap(sign_forward_func)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_sign_normal(context_mode):
    """
    Feature: Pyboost function.
    Description: Test function sign forward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = sign_forward_func(Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_ops_sign_vmap(context_mode):
    """
    Feature: Pyboost function.
    Description: Test function sign vmap forward.
    Expectation: Correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = sign_vmap_func(Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
def test_sign_dynamic():
    """
    Feature: Test sign with dynamic shape in graph mode using TEST_OP.
    Description: call ops.isclose with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((2, 3, 4), np.float32)
    TEST_OP(sign_forward_func, [[Tensor(x1)], [Tensor(x2)]], 'sign', disable_grad=True)
