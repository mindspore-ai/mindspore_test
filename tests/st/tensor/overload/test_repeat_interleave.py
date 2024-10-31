# Copyright 2023 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_repeat_interleave(mode):
    """
    Feature: Tensor.repeat_interleave
    Description: Verify the result of repeat_interleave.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    input_x = Tensor(np.array([1, 2, 3]), ms.int32)
    # test diff args when repeats type is int
    output1 = np.array([1, 1, 2, 2, 3, 3])
    assert np.allclose(input_x.repeat_interleave(2).asnumpy(), output1)
    assert np.allclose(input_x.repeat_interleave(2, dim=None).asnumpy(), output1)
    assert np.allclose(input_x.repeat_interleave(dim=None, repeats=2).asnumpy(), output1)

    input_y = Tensor(np.array([[1, 2], [3, 4]]), ms.int32)
    output2 = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    assert np.allclose(input_y.repeat_interleave(2).asnumpy(), output2)

    output3 = np.array([[1, 1, 1, 2, 2, 2],
                        [3, 3, 3, 4, 4, 4]])
    assert np.allclose(input_y.repeat_interleave(3, dim=1).asnumpy(), output3)

    # test diff args when repeats type is Tensor
    output4 = np.array([[1, 2],
                        [3, 4],
                        [3, 4]])
    assert np.allclose(input_y.repeat_interleave(Tensor(np.array([1, 2])), dim=0).asnumpy(), output4)
