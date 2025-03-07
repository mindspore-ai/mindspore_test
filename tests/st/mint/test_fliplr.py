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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
import mindspore.mint as mint
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def fliplr_func(x):
    return mint.fliplr(x)


@test_utils.run_with_cell
def fliplr_forward_func(x):
    return fliplr_func(x)


@test_utils.run_with_cell
def fliplr_backward_func(x):
    return ops.grad(fliplr_func, (0,))(x)


@arg_mark(plat_marks=[
    'platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'
],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_fliplr_normal(mode):
    """
    Feature: fliplr
    Description: Verify the result of fliplr
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.arange(8).reshape((2, 2, 2)))
    out = fliplr_forward_func(x)
    expect_out = np.array([[[2., 3.], [0., 1.]], [[6., 7.], [4., 5.]]])
    assert np.allclose(out.asnumpy(), expect_out)
