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
import numpy as np
import pytest

import mindspore as ms
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def unsqueeze_forward(x, dim):
    out = ms.mint.unsqueeze(x, dim)
    return out


@test_utils.run_with_cell
def unsqueeze_backward(x, dim):
    return ms.grad(unsqueeze_forward, (0,))(x, dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_mint_unsqueeze_normal(mode):
    """
    Feature: mint.unqueeze
    Description: Verify the result of mint.unqueeze
    Expectation: success
    """
    ms.set_context(mode=mode)
    ms.context.set_context(jit_level='O0')

    x = ms.Tensor(np.arange(2 * 3).reshape((2, 3)), dtype=ms.float32)
    dim = 0
    expect_out = np.array(np.arange(2 * 3).reshape((1, 2, 3)))

    out = unsqueeze_forward(x, dim)
    assert out.shape == expect_out.shape
    assert np.allclose(out.asnumpy(), expect_out)

    expect_grad = np.ones((2, 3)).astype(np.float32)
    grad = unsqueeze_backward(x, 0)
    assert grad.shape == expect_grad.shape
    assert np.allclose(grad.asnumpy(), expect_grad)
