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
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, Parameter
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=ms.GRAPH_MODE)

@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            P.AssignAdd()(x, y)
            return x

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 5


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_input_parameter():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            P.AssignAdd()(x, y)
            return x

    input_x = Parameter(ms.Tensor(2, dtype=ms.int32), name='input_x')
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert input_x == 5
    assert out == 5


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_sub_inplace_add():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            z = x - y
            P.AssignAdd()(x, y)
            return x, z

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out[0] == 5
    assert out[1] == -1


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_sub_inplace_add_add():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            z = x - y
            P.AssignAdd()(z, y)
            w = z + y
            return w

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 5


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_sub_inplace_add_add_twice():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            z = x - y
            z1 = x - y
            P.AssignAdd()(z, y)
            w = z + y
            return w, z1

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out[0] == 5
    assert out[1] == -1


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_sub_inplace_add_inplace_sub():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            z = x - y
            P.AssignAdd()(z, y)
            w = z + y
            P.AssignSub()(z, x)
            return w, z

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out[0] == 5
    assert out[1] == 0


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_func_sub():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def add_func(self, x1, y1):
            P.AssignAdd()(x1, y1)
            return x1

        def construct(self, x, y):
            if x < y:
                self.add_func(x, y)
            y = P.Sub()(y, x)
            return x, y

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out[0]:", out[0])
    print("out[1]:", out[1])
    assert out[0] == 5
    assert out[1] == -2


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_func_sub_control_flow():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def add_func(self, x1, y1):
            if x1 != y1:
                P.AssignAdd()(x1, y1)
            else:
                x1 = y1 - x1
            return x1

        def construct(self, x, y):
            if x < y:
                self.add_func(x, y)
            y = P.Sub()(x, y)
            return x, y

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out[0]:", out[0])
    print("out[1]:", out[1])
    assert out[0] == 5
    assert out[1] == 2


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_func_sub_control_flow_2():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def func(self, x1, y1):
            if x1 == y1:
                x1 = x1 - y1
            else:
                x1 = y1 - x1
            return x1

        def construct(self, x, y):
            if x < y:
                z = self.func(x, y)
                P.AssignAdd()(z, y)
            else:
                z = x + y
            y = P.Sub()(z, y)
            return y

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 1


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_sub_func_3():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def func(self, x1, y1):
            if x1 == y1:
                return x1 - y1
            return y1 - x1

        def construct(self, x, y):
            P.AssignAdd()(x, y)
            z = self.func(x, y)
            return z

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == -2


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_multi_inplace_ops():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def construct(self, x, y):
            if x > y:
                P.AssignAdd()(x, y)
            else:
                P.AssignSub()(y, x)
            return x, y

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out[0]:", out[0])
    print("out[1]:", out[1])
    assert out[0] == 2
    assert out[1] == 1


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_parameter():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor(1, dtype=ms.int32), name='param')

        def construct(self, x, y):
            P.AssignAdd()(x, y)
            self.param = x * 2 + y
            P.AssignSub()(x, y)
            return self.param, x

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out[0] == 13
    assert out[1] == 2


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_control_flow_multi():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor(1, dtype=ms.int32), name='param')

        def construct(self, x, y):
            P.AssignAdd()(x, y)
            if x * 2 > y:
                P.AssignAdd()(x, y - self.param)
            P.AssignSub()(y, x)
            return y

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == -4


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_add_control_flow_multi_2():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param = Parameter(Tensor(1, dtype=ms.int32), name='param')

        def construct(self, x, y):
            if x * 2 > y:
                P.AssignAdd()(x, y - self.param)
                z = x + y
                P.AssignSub()(z, self.param)
            else:
                z = x - y
            return z

    input_x = ms.Tensor(2, dtype=ms.int32)
    input_y = ms.Tensor(3, dtype=ms.int32)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out == 6


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_tensor_inplace_index_add():
    """
    Feature: Support tensor inplace.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.op = P.InplaceIndexAdd(axis=0)

        def construct(self, input_x, indices, updates):
            self.op(input_x, indices, updates)
            return input_x

    input_x = ms.Tensor([[1, 2], [3, 4], [5, 6]], dtype=ms.int32)
    indices = ms.Tensor([0, 1], dtype=ms.int32)
    updates = ms.Tensor([[1, 2], [7, 8]], dtype=ms.int32)
    net = Net()
    out = net(input_x, indices, updates)
    print("out:", out)
    assert (out.asnumpy() == [[2, 4], [10, 12], [5, 6]]).all()
