#Copyright 2020 Huawei Technologies Co., Ltd
#
#Licensed under the Apache License, Version 2.0(the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http:  // www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.
#== == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
from tests.mark_utils import arg_mark

import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore.common.api import jit
from mindspore.ops import functional as F
from mindspore import dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.select = P.Select()

    def construct(self, cond_op, input_x, input_y):
        return self.select(cond_op, input_x, input_y)


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_select_float32():
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[1.2, 1], [1, 0]]).astype(np.float32)
    y = np.array([[1, 2], [3, 4.0]]).astype(np.float32)
    select = Net()
    output = select(Tensor(cond), Tensor(x), Tensor(y))
    print(output.asnumpy())
    expect = [[1.2, 2], [1, 4.0]]
    error = np.ones(shape=[2, 2]) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_select_float16():
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[1.2, 1], [1, 0]]).astype(np.float16)
    y = np.array([[1, 2], [3, 4.0]]).astype(np.float16)
    select = Net()
    output = select(Tensor(cond), Tensor(x), Tensor(y))
    print(output.asnumpy())
    expect = [[1.2, 2], [1, 4.0]]
    error = np.ones(shape=[2, 2]) * 1.0e-3
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_select_int32():
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[12, 1], [1, 0]]).astype(np.int32)
    y = np.array([[1, 2], [3, 4]]).astype(np.int32)
    select = Net()
    output = select(Tensor(cond), Tensor(x), Tensor(y))
    print(output.asnumpy())
    expect = [[12, 2], [1, 4]]
    error = np.ones(shape=[2, 2]) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_functional_select_scalar():
    """
    Feature: Test functional select operator. Support x or y is a int/float.
    Description: Operator select's input `x` is a Tensor with int32 type, input `y` is a int.
    Expectation: Assert result.
    """
    cond = np.array([[True, False], [True, False]]).astype(np.bool)
    x = np.array([[12, 1], [1, 0]]).astype(np.int32)
    y = 2
    output = ops.select(Tensor(cond), Tensor(x), y)
    print(output.asnumpy())
    expect = [[12, 2], [1, 2]]
    error = np.ones(shape=[2, 2]) * 1.0e-6
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
    assert np.all(-diff < error)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_functional_select_broadcast():
    """
    Feature: Test functional select operator support broadcast input.
    Description: Operator select's support broadcast input.
    Expectation: Assert result.
    """
    cond = Tensor(np.random.rand(1, 65, 54, 12, 5, 2), dtype=mstype.bool_)
    x = Tensor(np.random.rand(5, 5, 65, 1, 12, 5, 2).astype(np.float32))
    y = Tensor(np.random.rand(65, 54, 1, 5, 2).astype(np.float32))

    @jit
    def foo(a, b, c):
        return F.select(a, b, c)

    ret = foo(cond, x, y)
    assert ret.shape == (5, 5, 65, 54, 12, 5, 2)
