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
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE])
def test_byte(mode):
    """
    Feature: Tensor.byte
    Description: Verify the result of byte.
    Expectation: expect correct result.
    """
    dtype_op = P.DType()
    a = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(a, dtype=ms.float32)
    ms.set_context(mode=mode)
    output = x.byte()
    assert x.shape == output.shape
    assert dtype_op(output) == ms.uint8
