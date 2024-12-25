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
import mindspore as ms
import mindspore.nn as nn
from mindspore import context
from tests.mark_utils import arg_mark

context.set_context(mode=ms.GRAPH_MODE)


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [SelectExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_select_ext_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[0] = y
            return x

    input_x = ms.Tensor([2], dtype=ms.int32)
    input_y = ms.Tensor([3], dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 3


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [SelectExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_select_ext_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0]
            return y

    input_x = ms.Tensor([2], dtype=ms.int32)
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert out == 2


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [SelectExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_slice_ext_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[0:2] = y
            return x

    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    input_y = ms.Tensor([3], dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert ms.ops.all(out == ms.Tensor([3, 3, 1], dtype=ms.int32))


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [SelectExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_slice_ext_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0:2]
            return y

    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [CopyExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_None_index_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[None] = y
            return x

    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    input_y = ms.Tensor([3], dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert ms.ops.all(out == ms.Tensor([3, 3, 3], dtype=ms.int32))


@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_None_index_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[None]
            return y

    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert ms.ops.all(out == ms.Tensor([1, 1, 1], dtype=ms.int32))


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [CopyExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_self_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[...] = y
            return x

    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    input_y = ms.Tensor([2], dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert ms.ops.all(out == ms.Tensor([2, 2, 2], dtype=ms.int32))


@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_self_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[...]
            return y

    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    net = Net()
    out = net(input_x)
    print("out:", out)
    assert ms.ops.all(out == ms.Tensor([1, 1, 1], dtype=ms.int32))


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [CopyExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_bool_index_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y, index):
            x[index] = y
            return x

    net = Net()
    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    input_y = ms.Tensor([2], dtype=ms.int32)
    index = True
    out1 = net(input_x, input_y, index)
    index = False
    out2 = net(input_x, input_y, index)
    print("index_True:", out1)
    print("index_False:", out2)
    assert ms.ops.all(out1 == ms.Tensor([2, 2, 2], dtype=ms.int32))
    assert ms.ops.all(out2 == ms.Tensor([1, 1, 1], dtype=ms.int32))


@pytest.mark.skip(
    reason="RuntimeError: Currently, the 'Index' supports only the pynative mode."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_bool_index_read_true():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, index):
            y = x[index]
            return y

    net = Net()
    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    index = True
    out1 = net(input_x, index)
    print("index_True:", out1)
    assert ms.ops.all(out1 == ms.Tensor([1, 1, 1], dtype=ms.int32))


@pytest.mark.skip(
    reason="RuntimeError: Currently, the 'Index' supports only the pynative mode."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_bool_index_read_false():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, index):
            y = x[index]
            return y

    net = Net()
    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    index = False
    out2 = net(input_x, index)
    print("index_False:", out2)
    assert ms.ops.all(out2 == ms.Tensor([], dtype=ms.int32))


@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_bool_tensor_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y, index):
            x[index] = y
            return x

    net = Net()
    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    input_y = ms.Tensor([2], dtype=ms.int32)
    index = ms.Tensor([True, False, True])
    out = net(input_x, input_y, index)
    print("out", out)
    assert ms.ops.all(out == ms.Tensor([2, 1, 2], dtype=ms.int32))


@pytest.mark.skip(
    reason="ValueError: For 'Equal', input1.shape and input2.shape need to broadcast."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_bool_tensor_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, index):
            y = x[index]
            return y

    net = Net()
    input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    index = ms.Tensor([True, False, True])
    out = net(input_x, index)
    print("out", out)  # mindspore: out = [1,1,1]; torch: out = [1,1]
    assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [SelectExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_select_slice_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[0, 0:2] = y
            return x

    net = Net()
    x = ms.Tensor([[1, 1, 1]], dtype=ms.int32)
    y = ms.Tensor([2], dtype=ms.int32)
    out = net(x, y)
    print("out", out)
    assert ms.ops.all(out == ms.Tensor([[2, 2, 1]], dtype=ms.int32))


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
def test_tensor_select_slice_write_2():
    """
    Feature: Support tensor inplace.
    Description: Tensor setitem by slice, then use this tensor to do some op.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x: ms.Tensor):
            x[1:] = 1
            return x + 1

    net = Net()
    x = ms.Tensor([[1, 2], [3, 4]])
    out = net(x)
    assert ms.ops.all(out == ms.Tensor([[2, 3], [2, 2]]))


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [SelectExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_select_slice_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0, 0:2]
            return y

    net = Net()
    x = ms.Tensor([[1, 1, 1]], dtype=ms.int32)
    out = net(x)
    print("out", out)
    assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))


@pytest.mark.skip(reason="NameError: name 'Tensor' is not defined.")
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_index_list_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[[0, 2]] = y
            return x

    net = Net()
    x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    y = ms.Tensor([2], dtype=ms.int32)
    out = net(x, y)
    print("out", out)
    assert ms.ops.all(out == ms.Tensor([2, 1, 2], dtype=ms.int32))


@pytest.mark.skip(reason="NameError: name 'Tensor' is not defined.")
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_index_list_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0, 2]
            return y

    net = Net()
    x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    out = net(x)
    print("out", out)
    assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))


@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_index_tensor_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y, index):
            x[index] = y
            return x

    net = Net()
    x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    y = ms.Tensor([2], dtype=ms.int32)
    index = ms.Tensor([0, 2], dtype=ms.int32)
    out = net(x, y, index)
    print("out", out)
    assert ms.ops.all(out == ms.Tensor([2, 1, 2], dtype=ms.int32))


@pytest.mark.skip(
    reason="ValueError: For 'Equal', input1.shape and input2.shape need to broadcast."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_index_tensor_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, index):
            y = x[index]
            return y

    net = Net()
    x = ms.Tensor([1, 1, 1], dtype=ms.int32)
    index = ms.Tensor([0, 2], dtype=ms.int32)
    out = net(x, index)
    print("out", out)  # mindspore: out = [1,1,1]; torch: out = [1,1]
    assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [SelectExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_select_slice_index_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[0, 0:1, [0, 2]] = y
            return x

    net = Net()
    x = ms.Tensor([[[1, 1, 1]]], dtype=ms.int32)
    y = ms.Tensor([2], dtype=ms.int32)
    out = net(x, y)
    print("out", out)
    assert ms.ops.all(out == ms.Tensor([[[2, 1, 2]]], dtype=ms.int32))


@pytest.mark.skip(
    reason="RuntimeError: Unsupported op [SelectExt] on GPU, \
                  Please confirm whether the device target setting is correct."
)
@arg_mark(
    plat_marks=["platform_gpu", "platform_ascend"],
    level_mark="level0",
    card_mark="onecard",
    essential_mark="essential",
)
def test_tensor_select_slice_index_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0, 0:1, [0, 2]]
            return y

    net = Net()
    x = ms.Tensor([[[1, 1, 1]]], dtype=ms.int32)
    out = net(x)
    print("out", out)
    assert ms.ops.all(out == ms.Tensor([[1, 1]], dtype=ms.int32))
