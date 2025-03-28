# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
import mindspore as ms
from mindspore import Tensor


@test_utils.run_with_cell
def reshape_as_forward_func(input_x, other):
    return input_x.reshape_as(other)


@test_utils.run_with_cell
def reshape_as_backward_func(input_x, other):
    return ms.grad(reshape_as_forward_func, (0,))(input_x, other)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu',
                      'platform_ascend', 'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_reshape_as(mode):
    """
    Feature: tensor.reshape_as
    Description: Verify the result of output
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = Tensor([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]], dtype=ms.float32)
    y = Tensor(np.arange(6).reshape(3, 2))

    forward_out = reshape_as_forward_func(x, y)
    expect_forward_output = np.array([[-0.1, 0.3],
                                      [3.6, 0.4],
                                      [0.5, -3.2]])
    assert np.allclose(forward_out.asnumpy(), expect_forward_output)

    backward_out = reshape_as_backward_func(x, y)
    expect_backward_output = np.array([[1, 1, 1],
                                       [1, 1, 1]])
    assert np.allclose(backward_out.asnumpy(), expect_backward_output)
