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
"""Tests for mint.narrow: input, dim, start, length."""
import numpy as np
import pytest

import mindspore as ms
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def narrow_forward(x, dim, start, length):
    out = ms.mint.narrow(x, dim, start, length)
    return out


@test_utils.run_with_cell
def narrow_backward(x, dim, start, length):
    return ms.grad(narrow_forward, (0,))(x, dim, start, length)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_narrow_normal(mode):
    """
    Feature: mint.narrow
    Description: Verify the result of mint.narrow
    Expectation: success
    """
    ms.set_context(mode=mode)
    ms.context.set_context(jit_level='O0')

    x_np = np.random.randn(2, 4).astype(np.float32)
    x = ms.Tensor(x_np)
    expect_out = x_np[:, 1:3]

    out = narrow_forward(x, -1, 1, 2)
    assert out.shape == expect_out.shape
    assert np.allclose(out.asnumpy(), expect_out)

    expect_grad = np.array([[0., 1., 1., 0.],
                            [0., 1., 1., 0.]]).astype(np.float32)
    grad = narrow_backward(x, -1, 1, 2)
    assert grad.shape == expect_grad.shape
    assert np.allclose(grad.asnumpy(), expect_grad)
