# Copyright 2022 Huawei Technologies Co., Ltd
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

import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore.ops as ops
from mindspore import context, Tensor
from mindspore.nn import Cell


class ReduceNet(Cell):
    def __init__(self):
        super(ReduceNet, self).__init__()
        self.add = ops.Add()
        self.sum = ops.ReduceSum(skip_mode=True)

    def construct(self, x, y, axis):
        z = self.add(x, y)
        return self.sum(z, axis)


class ReduceAxisNet(Cell):
    def __init__(self, axis):
        super(ReduceAxisNet, self).__init__()
        self.sum = ops.ReduceSum(keep_dims=False)
        self.axis = axis

    def construct(self, x):
        return self.sum(x, self.axis)


def run_case():
    np.random.seed(1)
    x = np.random.normal(0, 1, [1024, 1024]).astype(np.float32)
    y = np.random.normal(0, 1, [1024, 1024]).astype(np.float32)
    expect = np.sum(x + y, axis=(1,))

    net = ReduceNet()
    output = net(Tensor(x), Tensor(y), 1)
    assert np.allclose(expect, output.asnumpy(), 1.e-4, 1.e-4, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_case():
    """
    Feature: test reduce sum with skip_mode=True
    Description: dvm support ReduceSum with skip_mode=True when axis is not empty(ReduceSum won't be skipped)
    Expectation: the result match with numpy result
    """
    context.set_context(jit_level='O1')
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    run_case()


def run_reduce_with_axis(shape, axis):
    np.random.seed(1)
    x = np.random.normal(0, 1, shape).astype(np.float32)
    expect = np.sum(x, axis=axis, keepdims=False)

    net = ReduceAxisNet(axis)
    output = net(Tensor(x))
    assert np.allclose(expect, output.asnumpy(), 1.e-4, 1.e-4, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("shape, axis", [((1,), (0,)), ((1024, 1, 32), (1,))])
def test_reduce_arithmetic_simplify(shape, axis):
    """
    Feature: test reduce sum with enable_graph_kernel=True
    Description: reduce sum with Arithmetic Simplify
    Expectation: the result match with numpy result
    """
    context.set_context(jit_level='O0')
    context.set_context(mode=context.GRAPH_MODE, enable_graph_kernel=True)
    run_reduce_with_axis(shape, axis)
