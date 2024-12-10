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

import numpy as np
import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops


class Net_index3(nn.Cell):
    def construct(self, x, index1, index2, index3):
        y = x[index1, index2, index3]
        return y


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_getitem_index_negative(mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem with indexes are negative
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net_index3()
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    index1_np = -1
    index2_np = -1
    index3_np = -1
    y_np = x_np[index1_np, index2_np, index3_np]
    y = net(Tensor(x_np), Tensor(index1_np), Tensor(index2_np), Tensor(index3_np))
    assert np.allclose(y_np, y.asnumpy())


class Net_index2_slice(nn.Cell):
    def construct(self, x, index1, index2):
        y = x[index1, 0:2, index2]
        return y


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_getitem_index_negative_with_slice(mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem with indexes are negative
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net_index2_slice()
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    index1_np = -1
    index2_np = -1
    y_np = x_np[index1_np, 0:2, index2_np]
    y = net(Tensor(x_np), Tensor(index1_np), Tensor(index2_np))
    assert np.allclose(y_np, y.asnumpy())


def tensor_getitem_func(x):
    y0 = x[0]
    y1 = x[0:2]
    y2 = x[True]
    y3 = x[False]
    y4 = x[None]
    y5 = x[...]
    y6 = x[0:2, None, ...]
    y7 = x[[0, 1]]
    y8 = x[np.array(0)]
    y9 = x[np.array(True)]
    y10 = x[np.array(False)]
    y11 = x[np.array(0), np.array(1)]
    y12 = x[np.array([0, 1])]
    return y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE,])
def test_getitem_refactor(mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem with refactor codes
    Expectation: success
    """
    ms.set_context(mode=mode)
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    # basic index
    np_y = tensor_getitem_func(np_x)
    ms_y = tensor_getitem_func(ms_x)
    for i in range(len(np_y)):
        assert np.allclose(
            np_y[i], ms_y[i].asnumpy()), f"idx:{i}, np_y:{np_y[i]} {np_y[i].shape}, ms_y:{ms_y[i]} {ms_y[i].shape}"
    # tensor index
    np_y = np_x[0]
    ms_y = ms_x[Tensor(0)]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    np_y = np_x[True]
    ms_y = ms_x[Tensor(True)]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    np_y = np_x[False]
    ms_y = ms_x[Tensor(False)]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    np_y = np_x[0:2]
    ms_y = ms_x[Tensor(0):Tensor(2)]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    np_y = np_x[[0, 1]]
    ms_y = ms_x[Tensor([0, 1])]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    # tuple index
    np_y = np.array([[[0, 1, 2, 3]], [[4, 5, 6, 7]]])
    ms_y = ms_x[0, 0:2, True]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    np_y = np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]])
    ms_y = ms_x[[0, None, ...]]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    # fancy index
    np_y = np.array([[0, 1, 2, 3], [16, 17, 18, 19]])
    ms_y = ms_x[[0, 1], [0, 1]]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    ms_y = ms_x[Tensor([0, 1]), Tensor([0, 1])]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"

    np_y = np.array([0, 13])
    ms_y = ms_x[[0, 1], 0, [0, 1]]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    ms_y = ms_x[Tensor([0, 1]), Tensor(0), Tensor([0, 1])]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"

    np_y = np.array([0, 5])
    ms_y = ms_x[0, [0, 1], [0, 1]]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    ms_y = ms_x[Tensor(0), Tensor([0, 1]), Tensor([0, 1])]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    ms_y = ms_x[Tensor(0), [0, Tensor(1)], [Tensor(0), Tensor(1)]]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"

    np_y = np.array([[0, 4], [13, 17]])
    ms_y = ms_x[[0, 1], 0:2, [0, 1]]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    ms_y = ms_x[Tensor([0, 1]), 0:2, Tensor([0, 1])]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"

    np_y = np.array([[0, 1, 2, 3], [16, 17, 18, 19]])
    ms_y = ms_x[[0, 1], True, [0, 1]]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    ms_y = ms_x[Tensor([0, 1]), Tensor(True), Tensor([0, 1])]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"

    np_y = np.array([[[0, 1, 2, 3]], [[16, 17, 18, 19]]])
    ms_y = ms_x[[0, 1], None, [0, 1]]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    ms_y = ms_x[Tensor([0, 1]), None, Tensor([0, 1])]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"

    np_y = np.array([[0, 4, 8], [13, 17, 21]])
    ms_y = ms_x[[0, 1], ..., [0, 1]]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    ms_y = ms_x[Tensor([0, 1]), ..., Tensor([0, 1])]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE,])
def test_getitem_refactor_v2(mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem v2 with refactor cpp codes
    Expectation: success
    """
    ms.set_context(mode=mode)
    np_x = np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    ms_x = Tensor(np_x)

    np_y = np.array([[[60, 61],
                      [65, 66],
                      [70, 71],
                      [75, 76]],

                     [[80, 81],
                      [85, 86],
                      [90, 91],
                      [95, 96]],

                     [[100, 101],
                      [105, 106],
                      [110, 111],
                      [115, 116]]])
    ms_y = ms_x[1, ..., True, [0, 1]]
    assert np.allclose(np_y, ms_y.asnumpy()
                       ), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"
    ms_y = ms_x[Tensor(1), ..., Tensor(True), Tensor([0, 1])]
    assert np.allclose(np_y, ms_y.asnumpy()
                       ), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"

    np_y = np.array([[[[[0],
                        [1],
                        [2],
                        [3],
                        [4]],

                       [[5],
                        [6],
                        [7],
                        [8],
                        [9]],

                       [[10],
                        [11],
                        [12],
                        [13],
                        [14]],


                       [[15],
                        [16],
                        [17],
                        [18],
                        [19]]],


                      [[[20],
                        [21],
                        [22],
                        [23],
                        [24]],

                       [[25],
                        [26],
                        [27],
                        [28],
                        [29]],

                       [[30],
                        [31],
                        [32],
                        [33],
                        [34]],

                       [[35],
                        [36],
                        [37],
                        [38],
                        [39]]],


                      [[[40],
                        [41],
                        [42],
                        [43],
                        [44]],

                       [[45],
                        [46],
                        [47],
                        [48],
                        [49]],

                       [[50],
                        [51],
                        [52],
                        [53],
                        [54]],

                       [[55],
                        [56],
                        [57],
                        [58],
                        [59]]]]])
    ms_y = ms_x[0:1, ..., None]
    assert np.allclose(np_y, ms_y.asnumpy()), f"y:{np_y} {np_y.shape}, ms_y:{ms_y} {ms_y.shape}"


class Net_getitem_index_grad1(nn.Cell):
    def construct(self, x, index1, index2, index3):
        x = ops.relu(x)
        y = x[index1, index2, index3]
        return y


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE,])
def test_getitem_refactor_grad1(mode):
    """
    Feature: tensor getitem grad
    Description: Verify the result of tensor getitem grad1 with refactor cpp codes
    Expectation: success
    """
    ms.set_context(mode=mode)

    x_np = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    index1_np = -1
    index2_np = -1
    index3_np = -1

    pt_x_grad = [[[0., 0., 0., 0.,],
                  [0., 0., 0., 0.,],
                  [0., 0., 0., 0.,]],
                 [[0., 0., 0., 0.,],
                  [0., 0., 0., 0.,],
                  [0., 0., 0., 1.,]]]
    pt_x_grad_np = np.array(pt_x_grad)

    net = Net_getitem_index_grad1()
    ms_grad_func = ms.grad(net)
    ms_grad = ms_grad_func(Tensor(x_np), Tensor(index1_np), Tensor(index2_np), Tensor(index3_np))

    assert np.allclose(pt_x_grad_np, ms_grad.asnumpy()
                       ), f"pt_x_grad_np:{pt_x_grad_np} {pt_x_grad_np.shape}, ms_grads:{ms_grad} {ms_grad.shape}"


class Net_getitem_index_grad2(nn.Cell):
    def construct(self, x, i):
        x = ops.relu(x)
        # basic index
        if i == 0:
            ms_y = x[0]
        if i == 1:
            ms_y = x[0:2]
        if i == 2:
            ms_y = x[True]
        if i == 3:
            ms_y = x[False]
        if i == 4:
            ms_y = x[...]
        if i == 5:
            ms_y = x[None]
        if i == 6:
            ms_y = x[0:2, None, ...]
        if i == 7:
            ms_y = x[[0, 1]]
        # tensor index
        if i == 8:
            ms_y = x[Tensor(0)]
        if i == 9:
            ms_y = x[Tensor(True)]
        if i == 10:
            ms_y = x[Tensor(False)]
        if i == 11:
            ms_y = x[Tensor(0):Tensor(2)]
        if i == 12:
            ms_y = x[Tensor([0, 1])]
        # tuple index
        if i == 13:
            ms_y = x[0, 0:2, True]
        if i == 14:
            ms_y = x[[0, None, ...]]
        # fancy index
        if i == 15:
            ms_y = x[[0, 1], [0, 1]]
        if i == 16:
            ms_y = x[Tensor([0, 1]), Tensor([0, 1])]
        if i == 17:
            ms_y = x[[0, 1], 0, [0, 1]]
        if i == 18:
            ms_y = x[Tensor([0, 1]), Tensor(0), Tensor([0, 1])]
        if i == 19:
            ms_y = x[0, [0, 1], [0, 1]]
        if i == 20:
            ms_y = x[Tensor(0), Tensor([0, 1]), Tensor([0, 1])]
        if i == 21:
            ms_y = x[[0, 1], 0:2, [0, 1]]
        if i == 22:
            ms_y = x[Tensor([0, 1]), 0:2, Tensor([0, 1])]
        if i == 23:
            ms_y = x[[0, 1], True, [0, 1]]
        if i == 24:
            ms_y = x[Tensor([0, 1]), Tensor(True), Tensor([0, 1])]
        if i == 25:
            ms_y = x[[0, 1], None, [0, 1]]
        if i == 26:
            ms_y = x[Tensor([0, 1]), None, Tensor([0, 1])]
        if i == 27:
            ms_y = x[[0, 1], ..., [0, 1]]
        if i == 28:
            ms_y = x[Tensor([0, 1]), ..., Tensor([0, 1])]
        return ms_y


pt_y0 = [[[0., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]],
         [[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]]]
pt_y0_np = np.array(pt_y0)

pt_y1 = [[[0., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]],
         [[1., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]]]
pt_y1_np = np.array(pt_y1)

pt_y2 = [[[0., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]],
         [[1., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]]]
pt_y2_np = np.array(pt_y2)

pt_y3 = [[[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]],
         [[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]]]
pt_y3_np = np.array(pt_y3)

pt_y4 = [[[0., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]],
         [[1., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]]]
pt_y4_np = np.array(pt_y4)

pt_y5 = [[[0., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]],
         [[1., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]]]
pt_y5_np = np.array(pt_y5)

pt_y6 = [[[0., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]],
         [[1., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]]]
pt_y6_np = np.array(pt_y6)

pt_y7 = [[[0., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]],
         [[1., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]]]
pt_y7_np = np.array(pt_y7)

pt_y8 = [[[0., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]],
         [[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]]]
pt_y8_np = np.array(pt_y8)

pt_y9 = [[[0., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]],
         [[1., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]]]
pt_y9_np = np.array(pt_y9)

pt_y10 = [[[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y10_np = np.array(pt_y10)

pt_y11 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y11_np = np.array(pt_y11)

pt_y12 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y12_np = np.array(pt_y12)

pt_y13 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y13_np = np.array(pt_y13)

pt_y14 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y14_np = np.array(pt_y14)

pt_y15 = [[[0., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [0., 0., 0., 0.,]]]
pt_y15_np = np.array(pt_y15)

pt_y16 = [[[0., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [0., 0., 0., 0.,]]]
pt_y16_np = np.array(pt_y16)

pt_y17 = [[[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 1., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y17_np = np.array(pt_y17)

pt_y18 = [[[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 1., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y18_np = np.array(pt_y18)

pt_y19 = [[[0., 0., 0., 0.,],
           [0., 1., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y19_np = np.array(pt_y19)

pt_y20 = [[[0., 0., 0., 0.,],
           [0., 1., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y20_np = np.array(pt_y20)

pt_y21 = [[[0., 0., 0., 0.,],
           [1., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 1., 0., 0.,],
           [0., 1., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y21_np = np.array(pt_y21)

pt_y22 = [[[0., 0., 0., 0.,],
           [1., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 1., 0., 0.,],
           [0., 1., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y22_np = np.array(pt_y22)

pt_y23 = [[[0., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [0., 0., 0., 0.,]]]
pt_y23_np = np.array(pt_y23)

pt_y24 = [[[0., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [0., 0., 0., 0.,]]]
pt_y24_np = np.array(pt_y24)

pt_y25 = [[[0., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [0., 0., 0., 0.,]]]
pt_y25_np = np.array(pt_y25)

pt_y26 = [[[0., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [0., 0., 0., 0.,]]]
pt_y26_np = np.array(pt_y26)

pt_y27 = [[[0., 0., 0., 0.,],
           [1., 0., 0., 0.,],
           [1., 0., 0., 0.,]],
          [[0., 1., 0., 0.,],
           [0., 1., 0., 0.,],
           [0., 1., 0., 0.,]]]
pt_y27_np = np.array(pt_y27)

pt_y28 = [[[0., 0., 0., 0.,],
           [1., 0., 0., 0.,],
           [1., 0., 0., 0.,]],
          [[0., 1., 0., 0.,],
           [0., 1., 0., 0.,],
           [0., 1., 0., 0.,]]]
pt_y28_np = np.array(pt_y28)

pt_results = [pt_y0_np, pt_y1_np, pt_y2_np, pt_y3_np, pt_y4_np, pt_y5_np, pt_y6_np, pt_y7_np, pt_y8_np, pt_y9_np,
              pt_y10_np, pt_y11_np, pt_y12_np, pt_y13_np, pt_y14_np, pt_y15_np, pt_y16_np, pt_y17_np, pt_y18_np,
              pt_y19_np, pt_y20_np, pt_y21_np, pt_y22_np, pt_y23_np, pt_y24_np, pt_y25_np, pt_y26_np, pt_y27_np,
              pt_y28_np]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE,])
def test_getitem_refactor_grad2(mode):
    """
    Feature: tensor getitem grad
    Description: Verify the result of tensor getitem grad2 with refactor cpp codes
    Expectation: success
    """
    ms.set_context(mode=mode)

    k = len(pt_results)
    for i in range(k):
        x_np = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)

        net = Net_getitem_index_grad2()
        net.set_inputs()
        ms_grad_func = ms.grad(net)

        ms_x = Tensor(x_np)
        ms_grad = ms_grad_func(ms_x, i)

        err_msg = f"i:{i}, pt_x.grad:{pt_results[i]} {pt_results[i].shape}, ms_grads:{ms_grad} {ms_grad.shape}"
        assert np.allclose(pt_results[i], ms_grad.asnumpy()), err_msg


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE,])
def test_getitem_refactor_exception(mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem exception with refactor codes
    Expectation: success
    """
    ms.set_context(mode=mode)

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    with pytest.raises(IndexError) as exc:
        _ = ms_x[2, 0, 0]
    assert "is out of bounds for dimension" in str(exc.value)

    with pytest.raises(IndexError) as exc:
        _ = ms_x[0, 0, 0, 0]
    assert "too many indices for tensor with dimension size" in str(exc.value)

    with pytest.raises(IndexError) as exc:
        _ = ms_x[0, 't']
    assert "Invalid tensor index type" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        _ = ms_x[0:3:-1]
    assert "slice step must be positive" in str(exc.value)

    ms_x = Tensor(0)
    with pytest.raises(TypeError) as exc:
        _ = ms_x[0]
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    with pytest.raises(TypeError) as exc:
        _ = ms_x[0:1]
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    with pytest.raises(TypeError) as exc:
        _ = sum(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)
