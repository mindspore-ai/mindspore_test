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
# ==============================================================================
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.skip(reason="The operation is Unsupported")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_reshape():
    """
    Feature: Support tensor inplace view.
    Description: Support tensor inplace view.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x = x + 2
            reshape_x = P.Reshape()(x, (2,))
            reshape_x.add_(y)
            z = x + 1
            return z

    input_x = ms.Tensor([[2, 2]], dtype=ms.int32)
    input_y = ms.Tensor([3, 3], dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    assert (out.asnumpy() == [8, 8]).all()


@pytest.mark.skip(reason="The operation is Unsupported")
@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_split():
    """
    Feature: Support tensor inplace view.
    Description: Support tensor inplace view.
    Expectation: Run success.
    """
    class TensorSplitNet(nn.Cell):
        def __init__(self, axis=0, output_num=2):
            super(TensorSplitNet, self).__init__()
            self.split = P.Split(axis, output_num)

        def construct(self, x):
            x1, x2 = self.split(x)
            x1.add_(x2)
            x2.sub_(x1)
            y = x * 2
            return y

    np_x = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(np_x, dtype=ms.float32)
    out = TensorSplitNet(0, 2)(x)
    print("out:", out)
