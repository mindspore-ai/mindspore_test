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
from mindspore import Tensor, mint
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


@test_utils.run_with_cell
def ELU_forward(x, alpha):
    op = mint.nn.ELU(alpha)
    return op(x)


@test_utils.run_with_cell
def ELU_grad(x, alpha):
    op = mint.nn.ELU(alpha)
    return ms.grad(op, (0,))(x)

@test_utils.run_with_cell
def ELU_forward_for_dyn(x):
    op = mint.nn.ELU(1.0)
    return op(x)

def np_elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))


def np_elu_grad(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_elu_net(mode):
    """
    Feature: ELU
    Description: Verify the result of mint.nn.ELU network.
    Expectation: success
    """
    alpha = np.random.uniform(0.5, 2)
    x_np = np.random.randn(1, 2, 4, 4).astype(np.float32)
    x = Tensor(x_np, dtype=ms.float32)
    output = ELU_forward(x, alpha)
    expect_output_shape = (1, 2, 4, 4)
    expect_output = np_elu(x_np, alpha)
    expect_output_grad = np_elu_grad(x_np, alpha)
    assert np.allclose(expect_output_shape, output.shape)

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = ELU_forward(x, alpha)
        out_grad = ELU_grad(x, alpha)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        output = ELU_forward(x, alpha)
        out_grad = ELU_grad(x, alpha)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O2")
        output = ELU_forward(x, alpha)
        out_grad = ELU_grad(x, alpha)

    np.testing.assert_allclose(output.asnumpy(), expect_output, rtol=1e-4)
    np.testing.assert_allclose(out_grad.asnumpy(), expect_output_grad, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_mint_elu_dyn():
    """
    Feature: Dynamic shape of ELU.
    Description: test ELU with dynamic rank/shape.
    Expectation: success.
    """
    in1 = Tensor(np.random.randn(1, 2, 4, 4).astype(np.float32))
    in2 = Tensor(np.random.randn(2, 1, 4).astype(np.float32))
    TEST_OP(ELU_forward_for_dyn, [[in1], [in2]], '', disable_yaml_check=True)
