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
from mindspore import Tensor
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def take_along_dim_forward_func(x, indices, dim=None):
    return x.take_along_dim(indices, dim)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_take_along_dim_normal(mode):
    """
    Feature: tensor.take_along_dim
    Description: Verify the result of tensor.take_along_dim
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.context.set_context(jit_level='O0')
    np_x = generate_random_input((2, 3), np.float32)

    np_sort_indices = np.argsort(np_x, axis=1)

    expect_output = np.take_along_axis(np_x, np_sort_indices, axis=1)

    output = take_along_dim_forward_func(Tensor(np_x), Tensor(np_sort_indices), 1)
    assert np.allclose(output.asnumpy(), expect_output)
