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
import mindspore.common.dtype as mstype
from mindspore import ops, jit, JitConfig
from mindspore.ops import isfinite
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return np.isfinite(x)


def generate_expect_backward_output(x):
    return 0


@test_utils.run_with_cell
def isfinite_forward_func(x):
    return isfinite(x)


@test_utils.run_with_cell
def isfinite_backward_func(x):
    return ms.grad(isfinite_forward_func, (0))(x)


@test_utils.run_with_cell
def isfinite_vmap_func(x, in_axes=0):
    return ops.vmap(isfinite_forward_func, in_axes, out_axes=0)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_isfinite_normal(mode):
    """
    Feature: Test isfinite with static shape in graph and pynative mode.
    Description: call ops.isfinite with valid input and index.
    Expectation: return the correct value.1
    """
    x = generate_random_input((7168, 8192), np.float32)
    x1 = generate_random_input((8192, 7168), np.float32)

    if mode == 'pynative':
        output = isfinite_forward_func(ms.Tensor(x))
        output1 = isfinite_backward_func(ms.Tensor(x1))
    elif mode == 'KBK':
        output = (jit(isfinite_forward_func, jit_level="O0"))(ms.Tensor(x))
        output1 = (jit(isfinite_backward_func, jit_level="O0"))(ms.Tensor(x1))
    else:
        output = (jit(isfinite_forward_func, jit_level="O2"))(ms.Tensor(x))
        output1 = (jit(isfinite_backward_func, jit_level="O2"))(ms.Tensor(x1))

    expect = generate_expect_forward_output(x)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4)

    expect1 = generate_expect_backward_output(x1)
    assert np.allclose(output1.asnumpy(), expect1, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_isfinite_bfloat16(context_mode):
    """
    Feature: pyboost function.
    Description: test function isfinite forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2560, 8192), np.float32)
    output = isfinite_forward_func(ms.Tensor(x, mstype.bfloat16))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_isfinite_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function isfinite vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((8192, 2048), np.float32)
    output = isfinite_vmap_func(ms.Tensor(x), 0)
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_isfinite_dynamic_shape_testop():
    """
    Feature: Test isfinite with dynamic shape in graph mode using TEST_OP.
    Description: call ops.isfinite with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((3, 4, 5), np.float32)
    x2 = generate_random_input((3, 7, 8, 3), np.float32)

    TEST_OP(isfinite_forward_func, [[ms.Tensor(x1)], [ms.Tensor(x2)]], 'isfinite')
