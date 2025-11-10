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
"""Tests for mint.permute: input, dims."""
import numpy as np
import pytest

import mindspore as ms
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def permute_forward(x, dims):
    out = ms.mint.permute(x, dims)
    return out


@test_utils.run_with_cell
def permute_backward(x, dims):
    return ms.grad(permute_forward, (0,))(x, dims)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_permute_normal(mode):
    """
    Feature: mint.permute
    Description: Verify the result of mint.permute
    Expectation: success
    """
    ms.set_context(mode=mode)
    ms.context.set_context(jit_level='O0')

    x_np = np.random.randn(2, 3, 4).astype(np.float32)
    x = ms.Tensor(x_np)
    dims = (2, 0, 1)
    expect_out = x_np.transpose(dims)

    out = permute_forward(x, dims)
    assert out.shape == expect_out.shape
    assert np.allclose(out.asnumpy(), expect_out)

    expect_grad = np.ones((2, 3, 4)).astype(np.float32)
    grad = permute_backward(x, dims)
    assert grad.shape == expect_grad.shape
    assert np.allclose(grad.asnumpy(), expect_grad)
