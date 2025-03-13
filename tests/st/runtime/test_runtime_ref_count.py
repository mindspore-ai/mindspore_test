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
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, Tensor, mutable, jit, ops
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark

context.set_context(mode=ms.GRAPH_MODE, jit_config={"jit_level": "O0"})

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_no_used_input_from_copy_actor_to_super_kerenl_actor():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.add.set_device("CPU")
            self.depend = P.Depend()

        def construct(self, x, y):
            z1 = self.add(x, y)
            z2 = x - y
            return z1, z2

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out[0] == 5
    assert out[1] == -1


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_no_used_input_from_super_kernel_actor_to_super_kerenl_actor():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.add.set_device("CPU")
            self.depend = P.Depend()

        def construct(self, x, y):
            z1 = x * y
            z2 = self.add(z1, y)
            z3 = self.add(z2, y)
            z4 = z3 - y
            return z1, z2, z3, z4

    input_x = ms.Tensor(2)
    input_y = ms.Tensor(3)
    net = Net()
    out = net(input_x, input_y)
    print("out:", out)
    assert out[0] == 6
    assert out[1] == 9
    assert out[2] == 12
    assert out[3] == 9


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_no_used_input_from_parameter_store_to_super_kerenl_actor():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.add.set_device("CPU")

        def construct(self, x, y):
            z3 = y * y
            return x.shape, z3

    net = Net()
    x_dyn = Tensor(shape=[None], dtype=ms.int64)
    y_dyn = Tensor(shape=[None], dtype=ms.int64)
    net.set_inputs(x_dyn, y_dyn)
    input_x = ms.Tensor([2])
    input_y = ms.Tensor([3])
    out = net(input_x, input_y)
    print("out:", out)
    assert out[1] == 9


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_no_used_flag_parameter():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.add = P.Add()
            self.add.set_device("CPU")

        def construct(self, x):
            return x[1] + x[1]

    net = Net()
    input_x = ms.Tensor([2])
    input_y = ms.Tensor([3])
    input_all = mutable([input_x, input_y])
    out = net(input_all)
    print("out:", out)
    assert out == 6


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_no_used_flag_parameter_2():
    """
    Feature: Support runtime ref count.
    Description: Support tensor inplace.
    Expectation: Run success.
    """
    @jit
    def avg_pool_forward_func(x):
        return ops.AvgPool(kernel_size=2, strides=2, pad_mode="VALID", data_format="NCHW")(x)

    @jit
    def avg_pool_backward_func(x):
        return ops.grad(avg_pool_forward_func, (0,))(x)

    x = ms.Tensor(np.arange(1 * 3 * 3 * 4).reshape(1, 3, 3, 4), ms.float32)
    grads = avg_pool_backward_func(x)
    print(grads)
