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
def test_tensor_numpy(mode):
    """
    Feature: tensor.numpy
    Description: Verify the result of numpy
    Expectation: success
    """
    ms.set_context(mode=mode)
    t = Tensor([1, 2, 3])
    expect_output = [1, 2, 3]
    assert np.array_equal(t.numpy(), expect_output)
    assert np.array_equal(t.numpy(force=False), expect_output)
    assert np.array_equal(t.numpy(force=True), expect_output)
