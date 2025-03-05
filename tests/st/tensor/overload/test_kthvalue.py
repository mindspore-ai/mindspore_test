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
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, input_x, k, dim=-1, keepdim=False):
        return input_x.kthvalue(k, dim=dim, keepdim=keepdim)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_kthvalue(mode):
    """
    Feature: tensor.kthvalue
    Description: Verify the result of kthvalue
    Expectation: success
    """
    ms.set_context(mode=mode)
    if mode == ms.GRAPH_MODE:
        ms.set_context(jit_config={"jit_level": "O0"})
    x = Tensor(np.array([0.9041, 0.0196, -0.3108, -2.4423]), ms.float32)
    k = 2
    dim = 0
    keepdim = True
    net = Net()
    values, indices = net(x, k, dim, keepdim)
    expect_values, expect_indices = [-0.3108], [2]
    assert np.allclose(values.asnumpy(), expect_values)
    assert np.allclose(indices.asnumpy(), expect_indices)
