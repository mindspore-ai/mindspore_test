# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore import ops
from mindspore.ops import operations as P


def strided_slice(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')

    x = Tensor(np.arange(0, 2 * 3 * 4 * 5).reshape(2, 3, 4, 5).astype(nptype))
    y = P.StridedSlice()(x, (1, 0, 0, 2), (2, 2, 2, 4), (1, 1, 1, 1))
    expect = np.array([[[[62, 63],
                         [67, 68]],
                        [[82, 83],
                         [87, 88]]]]).astype(nptype)
    assert np.allclose(y.asnumpy(), expect)

    y = P.StridedSlice()(x, (1, 0, 0, 5), (2, 2, 2, 1), (1, 1, 1, -2))
    expect = np.array([[[[64, 62],
                         [69, 67]],
                        [[84, 82],
                         [89, 87]]]]).astype(nptype)
    assert np.allclose(y.asnumpy(), expect)

    y = P.StridedSlice()(x, (1, 0, 0, -1), (2, 2, 2, 1), (1, 1, 1, -1))
    expect = np.array([[[[64, 63, 62],
                         [69, 68, 67]],
                        [[84, 83, 82],
                         [89, 88, 87]]]]).astype(nptype)
    assert np.allclose(y.asnumpy(), expect)

    y = P.StridedSlice()(x, (1, 0, -1, -2), (2, 2, 0, -5), (1, 1, -1, -2))
    expect = np.array([[[[78, 76],
                         [73, 71],
                         [68, 66]],
                        [[98, 96],
                         [93, 91],
                         [88, 86]]]]).astype(nptype)
    assert np.allclose(y.asnumpy(), expect)

    op = P.StridedSlice(begin_mask=0b1000, end_mask=0b0010, ellipsis_mask=0b0100)
    y = op(x, (1, 0, 0, 2), (2, 2, 2, 4), (1, 1, 1, 1))
    expect = np.array([[[[60, 61, 62, 63],
                         [65, 66, 67, 68],
                         [70, 71, 72, 73],
                         [75, 76, 77, 78]],
                        [[80, 81, 82, 83],
                         [85, 86, 87, 88],
                         [90, 91, 92, 93],
                         [95, 96, 97, 98]],
                        [[100, 101, 102, 103],
                         [105, 106, 107, 108],
                         [110, 111, 112, 113],
                         [115, 116, 117, 118]]]]).astype(nptype)
    assert np.allclose(y.asnumpy(), expect)

    x = Tensor(np.arange(0, 3 * 4 * 5).reshape(3, 4, 5).astype(nptype))
    y = P.StridedSlice()(x, (1, 0, 0), (2, -3, 3), (1, 1, 3))
    expect = np.array([[[20]]]).astype(nptype)
    assert np.allclose(y.asnumpy(), expect)

    x_np = np.arange(0, 4 * 5).reshape(4, 5).astype(nptype)
    y = Tensor(x_np)[:, ::-1]
    expect = x_np[:, ::-1]
    assert np.allclose(y.asnumpy(), expect)

    x = Tensor(np.arange(0, 2 * 3 * 4 * 5 * 4 * 3 * 2).reshape(2, 3, 4, 5, 4, 3, 2).astype(nptype))
    y = P.StridedSlice()(x, (1, 0, 0, 2, 1, 2, 0), (2, 2, 2, 4, 2, 3, 2), (1, 1, 1, 1, 1, 1, 2))
    expect = np.array([[[[[[[1498.]]],
                          [[[1522.]]]],
                         [[[[1618.]]],
                          [[[1642.]]]]],
                        [[[[[1978.]]],
                          [[[2002.]]]],
                         [[[[2098.]]],
                          [[[2122.]]]]]]]).astype(nptype)
    assert np.allclose(y.asnumpy(), expect)

    x = Tensor(np.arange(0, 2 * 3 * 4 * 5 * 5 * 4 * 3 * 2).reshape(2, 3, 4, 5, 5, 4, 3, 2).astype(nptype))
    y = P.StridedSlice()(x, (1, 0, 0, 2, 2, 1, 2, 0), (2, 2, 2, 4, 4, 2, 3, 2), (1, 1, 1, 1, 2, 1, 1, 2))
    expect = np.array([[[[[[[[7498.]]]],
                          [[[[7618.]]]]],
                         [[[[[8098.]]]],
                          [[[[8218.]]]]]],
                        [[[[[[9898.]]]],
                          [[[[10018.]]]]],
                         [[[[[10498.]]]],
                          [[[[10618.]]]]]]]]).astype(nptype)
    assert np.allclose(y.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_float64():
    strided_slice(np.float64)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_float32():
    strided_slice(np.float32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_float16():
    strided_slice(np.float16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_int64():
    strided_slice(np.int64)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_int32():
    strided_slice(np.int32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_int16():
    strided_slice(np.int16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_int8():
    strided_slice(np.int8)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_uint64():
    strided_slice(np.uint64)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_uint32():
    strided_slice(np.uint32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_uint16():
    strided_slice(np.uint16)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_uint8():
    strided_slice(np.uint8)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_strided_slice_bool():
    strided_slice(np.bool_)
    x = Tensor(np.arange(0, 4 * 4 * 4).reshape(4, 4, 4).astype(np.float32))
    y = x[-8:, :8]
    expect = np.array([[[0., 1., 2., 3.],
                        [4., 5., 6., 7.],
                        [8., 9., 10., 11.],
                        [12., 13., 14., 15.]],

                       [[16., 17., 18., 19.],
                        [20., 21., 22., 23.],
                        [24., 25., 26., 27.],
                        [28., 29., 30., 31.]],

                       [[32., 33., 34., 35.],
                        [36., 37., 38., 39.],
                        [40., 41., 42., 43.],
                        [44., 45., 46., 47.]],

                       [[48., 49., 50., 51.],
                        [52., 53., 54., 55.],
                        [56., 57., 58., 59.],
                        [60., 61., 62., 63.]]])
    assert np.allclose(y.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("dtype",
                         [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                          np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_slice_functional_with_attr_int32(dtype):
    """
    Feature: Test strided_slice functional interface.
    Description: Test strided_slice functional interface with attr int32.
    Expectation: success.
    """
    x = Tensor(np.array([[[1., 1., 1.], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(dtype))
    begin = Tensor(np.array([2, 0, 0]).astype(np.int32))
    end = Tensor(np.array([3, 2, 3]).astype(np.int32))
    strides = Tensor(np.array([1, 1, 1]).astype(np.int32))
    output = ops.strided_slice(x, begin, end, strides)
    expect = np.array([[[5., 5., 5.],
                        [6., 7., 8.]]]).astype(dtype)
    assert (output.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("dtype",
                         [np.bool_, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                          np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_slice_functional_with_attr_int64(dtype):
    """
    Feature: Test strided_slice functional interface.
    Description: Test strided_slice functional interface with attr int64.
    Expectation: success.
    """
    x = Tensor(np.array([[[1., 1., 1.], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 7, 8]]]).astype(dtype))
    begin = Tensor(np.array([2, 0, 0]).astype(np.int64))
    end = Tensor(np.array([3, 2, 3]).astype(np.int64))
    strides = Tensor(np.array([1, 1, 1]).astype(np.int64))
    output = ops.strided_slice(x, begin, end, strides)
    expect = np.array([[[5., 5., 5.],
                        [6., 7., 8.]]]).astype(dtype)
    assert (output.asnumpy() == expect).all()
