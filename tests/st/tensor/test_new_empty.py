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
# pylint: disable=unused-variable

import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
import mindspore.common.dtype as mstype
from mindspore import nn
from mindspore import Tensor
from mindspore import context


class Net(nn.Cell):
    def construct(self, x, size, dtype, device):
        return x.new_empty(size, dtype=dtype, device=device)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [None, mstype.int32])
@pytest.mark.parametrize('device', [None, "CPU", "Ascend"])
def test_new_empty(mode, dtype, device):
    """
    Feature: tensor.new_empty()
    Description: Verify the result of tensor.new_empty
    Expectation: success
    """
    context.set_context(mode=mode)
    net = Net()
    size = (3, 3)
    x = Tensor(np.arange(4).reshape((2, 2)), dtype=mstype.float32)
    output = net(x, size, dtype, device)
    if dtype is None:
        assert output.dtype == mstype.float32
    else:
        assert output.dtype == dtype
    assert output.shape == size

def new_empty_forward_func(x):
    out = Net()(x, (2, 3, 4), None, None)
    return out.shape

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_new_empty_dynamic_shape():
    """
    Feature: dynamic shape forward features.
    Description: test new_empty forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = Tensor(np.arange(6).reshape(2, 3).astype(np.float32))
    tensor_x2 = Tensor(np.arange(60).reshape(3, 4, 5).astype(np.float32))

    TEST_OP(new_empty_forward_func, [[tensor_x1], [tensor_x2]], '', disable_yaml_check=True, disable_grad=True,
            disable_mode=['GRAPH_MODE', 'GRAPH_MODE_O0'])
