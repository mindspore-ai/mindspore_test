# Copyright 2024-2025 Huawei Technologies Co., Ltd
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
import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, mint, ops
from tests.mark_utils import arg_mark

context.set_context(jit_config={"jit_level": "O0"})


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([2], dtype=ms.int32)
        input_y = ms.Tensor([3], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x, input_y)
        print("out:", out)
        assert out == 3
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level0", card_mark="onecard", essential_mark="essential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([2], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x)
        print("out:", out)
        assert out == 2
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
def test_tensor_slice_ext_write():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            x[0: 2] = y
            return x

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        input_y = ms.Tensor([3], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x, input_y)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([3, 3, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        input_y = ms.Tensor([3], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x, input_y)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([3, 3, 3], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([1, 1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        input_y = ms.Tensor([2], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x, input_y)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([2, 2, 2], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        out = net(input_x)
        print("out:", out)
        assert ms.ops.all(out == ms.Tensor([1, 1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        input_y = ms.Tensor([2], dtype=ms.int32)
        index = True
        out1 = net(input_x, input_y, index)
        index = False
        out2 = net(input_x, input_y, index)
        print("index_True:", out1)
        print("index_False:", out2)
        assert ms.ops.all(out1 == ms.Tensor([2, 2, 2], dtype=ms.int32))
        assert ms.ops.all(out2 == ms.Tensor([2, 2, 2], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        index = True
        out1 = net(input_x, index)
        print("index_True:", out1)
        assert ms.ops.all(out1 == ms.Tensor([1, 1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        index = 0
        out2 = net(input_x, index)
        print("out2:", out2)
        assert ms.ops.all(out2 == ms.Tensor([], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        input_y = ms.Tensor([2], dtype=ms.int32)
        index = ms.Tensor([True, False, True])
        out = net(input_x, input_y, index)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([2, 1, 2], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        input_x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        index = ms.Tensor([True, False, True])
        out = net(input_x, index)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([[1, 1, 1]], dtype=ms.int32)
        y = ms.Tensor([2], dtype=ms.int32)
        out = net(x, y)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([[2, 2, 1]], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([[1, 2], [3, 4]])
        out = net(x)
        assert ms.ops.all(out == ms.Tensor([[2, 3], [2, 2]]))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
def test_tensor_select_slice_write_3():
    """
    Feature: Support tensor inplace.
    Description: Tensor setitem by slice, then use this tensor to do some op.
    Expectation: Run success; CSE pass shouldn't eliminate the second zeros_like node.
    """

    class Net(nn.Cell):
        def construct(self, x: ms.Tensor):
            a = mint.zeros_like(x)  # AbstractRefTensor-1
            a[1:] = 1
            b = a + 1
            a = mint.zeros_like(x)  # AbstractRefTensor-2, shouldn't be eliminated by CSE pass.
            a[0:1] = 2
            c = a * 2
            return b + c

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ops.randn(2, 2)
        result = net(x)
        expect = ms.Tensor([[5., 5.], [2., 2.]], dtype=ms.float32)
        assert ops.all(expect == result)
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([[1, 1, 1]], dtype=ms.int32)
        out = net(x)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        y = ms.Tensor([2], dtype=ms.int32)
        out = net(x, y)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([2, 1, 2], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
def test_tensor_index_list_read():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x):
            y = x[0: 2]
            return y

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        out = net(x)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        y = ms.Tensor([2], dtype=ms.int32)
        index = ms.Tensor([0, 2], dtype=ms.int32)
        out = net(x, y, index)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([2, 1, 2], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([1, 1, 1], dtype=ms.int32)
        index = ms.Tensor([0, 2], dtype=ms.int32)
        out = net(x, index)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([1, 1], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([[[1, 1, 1]]], dtype=ms.int32)
        y = ms.Tensor([2], dtype=ms.int32)
        out = net(x, y)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([[[2, 1, 2]]], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
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

    try:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
        net = Net()
        net.construct = ms.jit(net.construct, backend="ms_backend")
        x = ms.Tensor([[[1, 1, 1]]], dtype=ms.int32)
        out = net(x)
        print("out", out)
        assert ms.ops.all(out == ms.Tensor([[1, 1]], dtype=ms.int32))
    finally:
        del os.environ["MS_DEV_TENSOR_INDEX_BOOST"]
