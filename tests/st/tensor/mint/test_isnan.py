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
import os
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import Tensor

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_tensor_isnan(mode):
    """
    Feature: tensor.isnan
    Description: Verify the result of isnan
    Expectation: success
    """
    os.environ['MS_TENSOR_API_ENABLE_MINT'] = '1'
    ms.set_context(mode=mode)
    t = Tensor([1, float('nan'), 2])
    output = t.isnan()
    except_output = [False, True, False]
    assert np.array_equal(output.asnumpy(), except_output)
    del os.environ['MS_TENSOR_API_ENABLE_MINT']
