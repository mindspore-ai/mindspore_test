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
import numpy as np
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import Tensor
from tests.mark_utils import arg_mark

class Net(ms.nn.Cell):
    def construct(self, x):
        return x.pin_memory()

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
def test_tensor_pin_memory():
    """
    Feature: copy tensor to pin memory.
    Description: test tensor.pin_memory() with a cpu tensor.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    x = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)
    net = Net()
    y = net(x)
    assert y.is_pinned()
    np.allclose(y, x, rtol=1e-5, equal_nan=True)
