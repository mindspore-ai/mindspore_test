# Copyright 2020-2022 Huawei Technologies Co., Ltd
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
from mindspore.common import dtype as mstype
from mindspore.common.api import jit
from mindspore.ops.operations import _grad_ops as G
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class SliceGrad(nn.Cell):
    def __init__(self):
        super(SliceGrad, self).__init__()
        self.slice_grad = G.SliceGrad()

    @jit
    def construct(self, dy, x):
        return self.slice_grad(dy, x, (0, 1, 0), (2, 1, 3))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_slice_grad():
    x = Tensor(np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]]), mstype.float32)
    dy = Tensor(np.array([[[3., 1., 2.]], [[4., 1., 4.]]]), mstype.float32)
    slice_grad = SliceGrad()
    output = slice_grad(dy, x)
    expect = [[[0., 0., 0.],
               [3., 1., 2.]],
              [[0., 0., 0.],
               [4., 1., 4.]],
              [[0., 0., 0.],
               [0., 0., 0.]]]
    print("output:\n", output)
    assert (output.asnumpy() == expect).all()


class SliceGrad2(nn.Cell):
    def __init__(self):
        super(SliceGrad2, self).__init__()
        self.slice_grad = G.SliceGrad()

    def construct(self, dy, x):
        return self.slice_grad(dy, x, (0, 1, 0), (2, 2, 2))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_slice_grad2():
    dy = Tensor(np.array([[[2., 3.], [4., 5.]], [[8., 9.], [10., 11.]]]), mstype.float32)
    x = Tensor(np.arange(2 * 3 * 2).reshape(2, 3, 2), mstype.float32)
    grad = SliceGrad2()
    output = grad(dy, x)
    print("output:\n", output)
    expect = [[[0., 0.], [2., 3.], [4., 5.]],
              [[0., 0.], [8., 9.], [10., 11.]]]
    assert (output.asnumpy() == expect).all()


def test_slice_grad3():
    x = Tensor(np.array([[[1.0, 3.5, 5.8], [2.5, 4, 1]], [[3.5, 15.3, 3.1], [2.2, 4.0, 1.1]],
                         [[43.4, 1.1, 12.1], [2.4, 6.5, 6.3]]]), mstype.float64)
    dy = Tensor(np.array([[[3.1, 1.1, 2.2]], [[4.4, 1.2, 4.2]]]), mstype.float64)
    slice_grad = SliceGrad()
    output = slice_grad(dy, x)
    expect = [[[0., 0., 0.],
               [3.1, 1.1, 2.2]],
              [[0., 0., 0.],
               [4.4, 1.2, 4.2]],
              [[0., 0., 0.],
               [0., 0., 0.]]]
    print("output:\n", output)
    assert (output.asnumpy() == expect).all()


class SliceGrad8D(nn.Cell):
    def __init__(self):
        super(SliceGrad8D, self).__init__()
        self.slice_grad = G.SliceGrad()

    @jit
    def construct(self, dy, x):
        return self.slice_grad(dy, x, (1, 0, 2, 0, 0, 0, 0, 0), (1, 2, 1, 1, 1, 1, 1, 2))


def test_slice_grad_8d():
    """
    Feature: SliceGrad
    Description: test SliceGrad with 8D input
    Expectation: the output is as expected
    """
    x = Tensor(np.array([[[[[[[[6, 5]]]]], [[[[[4, 1]]]]], [[[[[7, 2]]]]]],
                          [[[[[[1, 5]]]]], [[[[[4, 8]]]]], [[[[[7, 5]]]]]]],
                         [[[[[[[4, 8]]]]], [[[[[1, 8]]]]], [[[[[0, 0]]]]]],
                          [[[[[[4, 8]]]]], [[[[[3, 3]]]]], [[[[[3, 9]]]]]]]]), mstype.int32)
    dy = Tensor(np.arange(1 * 2 * 1 * 1 * 1 * 1 * 1 * 2).reshape(1, 2, 1, 1, 1, 1, 1, 2), mstype.int32)
    slice_grad = SliceGrad8D()
    output = slice_grad(dy, x)
    expect = np.zeros((2, 2, 3, 1, 1, 1, 1, 2))
    expect[1:2, 0:2, 2:3, 0:1, 0:1, 0:1, 0:1, 0:2] = dy
    print("output:\n", output)
    assert (output.asnumpy() == expect).all()


class StridedSliceGrad(nn.Cell):
    def __init__(self, x, begin, end, stride):
        super(StridedSliceGrad, self).__init__()
        self.shape_op = P.Shape()
        self.shapex = self.shape_op(x)
        self.begin = begin
        self.end = end
        self.stride = stride
        self.stride_slice = G.StridedSliceGrad()

    def construct(self, dy):
        return self.stride_slice(dy, self.shapex, self.begin, self.end, self.stride)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_strided_slice_grad_bool_type():
    x = Tensor([[[False, False, True], [False, True, False]], [[False, True, False], [True, False, False]],
                [[False, True, True], [True, False, True]]], mstype.bool_)
    dy = Tensor([False, True, False], mstype.bool_)
    begin = (1, 0, 0)
    end = (2, 1, 3)
    stride = (1, 1, 1)
    slice_op = StridedSliceGrad(x, begin, end, stride)
    output = slice_op(dy)
    expected_output = np.array([[[False, False, False], [False, False, False]],
                                [[False, True, False], [False, False, False]],
                                [[False, False, False], [False, False, False]]])
    assert (output.asnumpy() == expected_output).all()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_strided_slice_grad_float32_type():
    x = Tensor([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]], mstype.float32)
    dy = Tensor([3, 3, 3], mstype.float32)
    begin = (1, 0, 0)
    end = (2, 1, 3)
    stride = (1, 1, 1)
    slice_op = StridedSliceGrad(x, begin, end, stride)
    output = slice_op(dy)
    expected_output = np.array([[[0, 0, 0], [0, 0, 0]], [[3, 3, 3], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]])
    assert (output.asnumpy() == expected_output).all()
