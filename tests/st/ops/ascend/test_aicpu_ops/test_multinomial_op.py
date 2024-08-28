# Copyright 2023 Huawei Technologies Co., Ltd
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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common.api import _pynative_executor

context.set_context(mode=context.GRAPH_MODE, device_target='Ascend')


class Net(nn.Cell):
    def __init__(self, sample):
        super(Net, self).__init__()
        self.sample = sample
        self.multinomial = P.Multinomial()

    def construct(self, x):
        return self.multinomial(x, self.sample)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_multinomial_1d(mode):
    """
    Feature: test Multinomial op.
    Description: test Multinomial op.
    Expectation: success.
    """
    context.set_context(mode=mode, device_target='Ascend')
    x = Tensor(np.array([0, 10, 0]).astype(np.float32))
    net = Net(1)
    out = net(x)
    expect_result = np.array([1]).astype(np.int32)
    np.array_equal(expect_result, out.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_multinomial_2d():
    """
    Feature: test Multinomial op.
    Description: test Multinomial op.
    Expectation: success.
    """
    x0 = Tensor(np.array([[2, 0], [0, 9]]).astype(np.float32))
    x1 = Tensor(np.array([[0, 0.1, 0], [0, 0, 1000]]).astype(np.float32))
    net0 = Net(1)
    net1 = Net(6)
    out0 = net0(x0)
    out1 = net1(x1)
    expect_result0 = np.array([[0], [1]]).astype(np.int32)
    expect_result1 = np.array([[1, 1, 1, 1, 1, 1], [2, 2, 2, 2, 2, 2]]).astype(np.int32)
    np.array_equal(expect_result0, out0.asnumpy())
    np.array_equal(expect_result1, out1.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_multinomial_0d():
    """
    Feature: test Multinomial op.
    Description: test Multinomial op.
    Expectation: success.
    """
    with pytest.raises(ValueError):
        x = Tensor(1)
        net = Net(1)
        net(x)
        _pynative_executor.sync()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_multinomial_3d():
    """
    Feature: test Multinomial op.
    Description: test Multinomial op.
    Expectation: success.
    """
    with pytest.raises(ValueError):
        x = Tensor(np.array([[[2, 0], [0, 9]]]).astype(np.float32))
        net = Net(1)
        net(x)
        _pynative_executor.sync()
