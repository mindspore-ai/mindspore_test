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
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_round(mode):
    """
    Feature: tensor.round
    Description: Verify the result of round
    Expectation: success
    """
    ms.set_context(mode=mode)
    input1 = Tensor(np.array([0.8, 1.5, 2.3, 2.5, -4.5]), ms.float32)
    expect_out1 = [1., 2., 2., 2., -4.]
    assert np.array_equal(input1.round().numpy(), expect_out1)
    input2 = Tensor(np.array([0.81, 1.52, 2.35, 2.53, -4.57]), ms.float32)
    expect_out2 = np.array([0.8, 1.5, 2.4, 2.5, -4.6], dtype='float32')
    assert np.array_equal(input2.round(decimals=1).numpy(), expect_out2)
