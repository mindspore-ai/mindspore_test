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

from tests.mark_utils import arg_mark
from mindspore import context
import mindspore as ms
import mindspore.nn as nn

context.set_context(mode=context.PYNATIVE_MODE)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.i = 1

    def construct(self, x, index=None):
        if index is None:
            return x.item()
        return x.item(index)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
def test_tensor_item():
    """
    Feature: tensor.item
    Description: Verify the result of item
    Expectation: success
    """
    net = Net()
    # ============== dtype of tensor is int8 ===========
    x = ms.Tensor(1, ms.int8)
    output = net(x)
    assert output == 1

    x = ms.Tensor([1, 2, 3], ms.int8)
    index = 1
    output = net(x, index)
    assert output == 2

    x = ms.Tensor([[1, 2], [3, 4]], ms.int8)
    index = (1, 0)
    output = net(x, index)
    assert output == 3

    # ============== dtype of tensor is uint8 ===========
    x = ms.Tensor(1, ms.uint8)
    output = net(x)
    assert output == 1

    x = ms.Tensor([1, 2, 3], ms.uint8)
    index = 1
    output = net(x, index)
    assert output == 2

    x = ms.Tensor([[1, 2], [3, 4]], ms.uint8)
    index = (1, 1)
    output = net(x, index)
    assert output == 4

    # ============== dtype of tensor is int16 ===========
    x = ms.Tensor(1, ms.int16)
    output = net(x)
    assert output == 1

    x = ms.Tensor([1, 2, 3], ms.int16)
    index = 1
    output = net(x, index)
    assert output == 2

    x = ms.Tensor([[1, 2], [3, 4]], ms.int16)
    index = (0, 1)
    output = net(x, index)
    assert output == 2

    # ============== dtype of tensor is uint16 ===========
    x = ms.Tensor(1, ms.uint16)
    output = net(x)
    assert output == 1

    x = ms.Tensor([1, 2, 3], ms.uint16)
    index = 1
    output = net(x, index)
    assert output == 2

    x = ms.Tensor([[1, 2], [3, 4]], ms.uint16)
    index = (1, 1)
    output = net(x, index)
    assert output == 4

    # ============== dtype of tensor is int32 ===========
    x = ms.Tensor(1, ms.int32)
    output = net(x)
    assert output == 1

    x = ms.Tensor([1, 2, 3], ms.int32)
    index = 1
    output = net(x, index)
    assert output == 2

    x = ms.Tensor([[1, 2], [3, 4]], ms.int32)
    index = (1, 1)
    output = net(x, index)
    assert output == 4

    # ============== dtype of tensor is uint32 ===========
    x = ms.Tensor(1, ms.uint32)
    output = net(x)
    assert output == 1

    x = ms.Tensor([1, 2, 3], ms.uint32)
    index = 1
    output = net(x, index)
    assert output == 2

    x = ms.Tensor([[1, 2], [3, 4]], ms.uint32)
    index = (1, 1)
    output = net(x, index)
    assert output == 4

    # ============== dtype of tensor is int64 ===========
    x = ms.Tensor(1, ms.int64)
    output = net(x)
    assert output == 1

    x = ms.Tensor([1, 2, 3], ms.int64)
    index = 1
    output = net(x, index)
    assert output == 2

    x = ms.Tensor([[1, 2], [3, 4]], ms.int64)
    index = (1, 1)
    output = net(x, index)
    assert output == 4

    # ============== dtype of tensor is uint64 ===========
    x = ms.Tensor(1, ms.uint64)
    output = net(x)
    assert output == 1

    x = ms.Tensor([1, 2, 3], ms.uint64)
    index = 1
    output = net(x, index)
    assert output == 2

    x = ms.Tensor([[1, 2], [3, 4]], ms.uint64)
    index = (1, 1)
    output = net(x, index)
    assert output == 4

    # ============== dtype of tensor is float32 ===========
    x = ms.Tensor(1.0, ms.float32)
    output = net(x)
    assert output == 1.0

    x = ms.Tensor([1.0, 2.0, 3.0], ms.float32)
    index = 1
    output = net(x, index)
    assert output == 2.0

    x = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float32)
    index = (1, 1)
    output = net(x, index)
    assert output == 4

    # ============== dtype of tensor is float64 ===========
    x = ms.Tensor(1.0, ms.float64)
    output = net(x)
    assert output == 1.0

    x = ms.Tensor([1.0, 2.0, 3.0], ms.float64)
    index = 1
    output = net(x, index)
    assert output == 2.0

    x = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.float64)
    index = (1, 1)
    output = net(x, index)
    assert output == 4

    # ============== dtype of tensor is bfloat16 ===========
    x = ms.Tensor(1.0, ms.bfloat16)
    output = net(x)
    assert output == x.asnumpy().item(0)

    x = ms.Tensor([1.0, 2.0, 3.0], ms.bfloat16)
    index = 1
    output = net(x, index)
    assert output == x.asnumpy().item(index)

    x = ms.Tensor([[1.0, 2.0], [3.0, 4.0]], ms.bfloat16)
    index = (1, 1)
    output = net(x, index)
    assert output == x.asnumpy().item(index)
