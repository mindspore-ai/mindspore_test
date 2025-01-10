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
from mindspore import mint
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(x, dim=None, correction=1, keepdim=False):
    expect_var = np.var(x, axis=dim, ddof=correction, keepdims=keepdim)
    expect_mean = np.mean(x, axis=dim, keepdims=keepdim)
    return expect_var, expect_mean

@test_utils.run_with_cell
def var_mean_forward_func(x, dim=None, correction=1, keepdim=False):
    return mint.var_mean(x, dim, correction=correction, keepdim=keepdim)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_var_mean_normal(mode):
    """
    Feature: mint.var_mean
    Description: Verify the result of mint.var_mean
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.context.set_context(jit_level='O0')
    x = generate_random_input((1, 2, 3), np.float32)

    expect_var, expect_mean = generate_expect_forward_output(x, dim=None, correction=1, keepdim=True)
    output = var_mean_forward_func(ms.Tensor(x), dim=None, correction=1, keepdim=True)

    assert np.allclose(output[0].asnumpy(), expect_var)
    assert np.allclose(output[1].asnumpy(), expect_mean)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_mint_var_mean_dynamic():
    """
    Feature: Test var_mean with dynamic shape in graph mode using TEST_OP.
    Description: call mint.var_mean with valid input and index.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((1, 2, 3), np.float32)
    dim1 = 0
    correction1 = 1
    keepdim1 = False

    x2 = generate_random_input((2, 3), np.float32)
    dim2 = 1
    correction2 = 0
    keepdim2 = True

    TEST_OP(var_mean_forward_func,
            [[ms.Tensor(x1), dim1, correction1, keepdim1], [ms.Tensor(x2), dim2, correction2, keepdim2]],
            "var_mean", disable_mode=["GRAPH_MODE"])
