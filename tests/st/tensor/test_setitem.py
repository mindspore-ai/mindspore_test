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
import copy

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops


class NetIndexBool(nn.Cell):
    def construct(self, x):
        x[True] *= 3
        return x


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_setitem_index_bool(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with indexes are bool
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = NetIndexBool()
    x = Tensor(np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32))
    x = net(Tensor(x))
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    x_np[True] *= 3
    assert np.allclose(x_np, x.asnumpy())


class NetIndexNone(nn.Cell):
    def construct(self, x):
        x[None] *= 3
        return x


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_setitem_index_none(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with indexes are None
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = NetIndexBool()
    x = Tensor(np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32))
    x = net(Tensor(x))
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    x_np[None] *= 3
    assert np.allclose(x_np, x.asnumpy())


def do_copy(x):
    if isinstance(x, Tensor):
        return x.copy()
    return copy.deepcopy(x)

def tensor_setitem_func(x):
    y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14 = \
        do_copy(x), do_copy(x), do_copy(x), do_copy(x), do_copy(x), do_copy(x), do_copy(x), do_copy(x), do_copy(x), \
        do_copy(x), do_copy(x), do_copy(x), do_copy(x), do_copy(x), do_copy(x)
    y0[0] = -1
    y1[0:2] = -1
    y2[True] = -1
    y3[False] = -1
    y4[None] = -1
    y5[...] = -1
    y6[0, 0:2, True] = -1
    y7[0:2, None, ...] = -1
    y8[[0, 1]] = -1
    y9[[0, 1]] = -1.0
    y10[np.array(0)] = -1
    y11[np.array(True)] = -1
    y12[np.array(False)] = -1
    y13[np.array(0), np.array(1)] = -1
    y14[np.array([0, 1])] = -1
    return y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12, y13, y14


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE,])
def test_setitem_refactor(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with refactor codes
    Expectation: success
    """
    ms.set_context(mode=mode)
    np_x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    # basic index
    np_y = tensor_setitem_func(np_x)
    ms_y = tensor_setitem_func(ms_x)
    for i in range(len(np_y)):
        assert np.allclose(
            np_y[i], ms_y[i].asnumpy()), f"idx:{i}, np_y:{np_y[i]} {np_y[i].shape}, ms_y:{ms_y[i]} {ms_y[i].shape}"
    # tensor index
    np_x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    np_x[0] = -1
    ms_x[Tensor(0)] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    np_x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    np_x[True] = -1
    ms_x[Tensor(True)] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    np_x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    np_x[False] = -1
    ms_x[Tensor(False)] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    np_x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    np_x[0:2] = -1
    ms_x[Tensor(0):Tensor(2)] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    np_x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    np_x[[0, 1]] = -1
    ms_x[Tensor([0, 1])] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    # tuple index
    np_x = np.array([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                     [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, None, ...]] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    # fancy index
    np_x = np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                     [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], [0, 1]] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), Tensor([0, 1])] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                     [[12, -1, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], 0, [0, 1]] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), Tensor(0), Tensor([0, 1])] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, 1, 2, 3], [4, -1, 6, 7], [8, 9, 10, 11]],
                     [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[0, [0, 1], [0, 1]] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor(0), Tensor([0, 1]), Tensor([0, 1])] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, 1, 2, 3], [-1, 5, 6, 7], [8, 9, 10, 11]],
                     [[12, -1, 14, 15], [16, -1, 18, 19], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], 0:2, [0, 1]] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), 0:2, Tensor([0, 1])] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                     [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], True, [0, 1]] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), Tensor(True), Tensor([0, 1])] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                     [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], None, [0, 1]] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), None, Tensor([0, 1])] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, 1, 2, 3], [-1, 5, 6, 7], [-1, 9, 10, 11]],
                     [[12, -1, 14, 15], [16, -1, 18, 19], [20, -1, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], ..., [0, 1]] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), ..., Tensor([0, 1])] = -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"


def tensor_setitem_with_add_func(x):
    y0, y1, y2, y3, y4, y5, y6, y7, y8 = do_copy(x), do_copy(x), do_copy(x), do_copy(x), do_copy(x), do_copy(x), \
                                         do_copy(x), do_copy(x), do_copy(x)
    y0[0] += -1
    y1[0:2] += -1
    y2[True] += -1
    y3[False] += -1
    y4[None] += -1
    y5[...] += -1
    y6[0, 0:2, True] += -1
    y7[0:2, None, ...] += -1
    y8[[0, 1]] += -1
    return y0, y1, y2, y3, y4, y5, y6, y7, y8


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE,])
def test_setitem_with_add_refactor(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with add with refactor codes
    Expectation: success
    """
    ms.set_context(mode=mode)
    np_x = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    np_y = tensor_setitem_with_add_func(np_x)
    ms_y = tensor_setitem_with_add_func(ms_x)
    for i in range(len(np_y)):
        assert np.allclose(
            np_y[i], ms_y[i].asnumpy()), f"idx:{i}, np_y:{np_y[i]} {np_y[i].shape}, ms_y:{ms_y[i]} {ms_y[i].shape}"

    # tuple index
    np_x = np.array([[[-1, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]],
                     [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, None, ...]] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    # fancy index
    np_x = np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                     [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], [0, 1]] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), Tensor([0, 1])] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                     [[12, 12, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], 0, [0, 1]] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), Tensor(0), Tensor([0, 1])] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, 1, 2, 3], [4, 4, 6, 7], [8, 9, 10, 11]],
                     [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[0, [0, 1], [0, 1]] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor(0), Tensor([0, 1]), Tensor([0, 1])] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, 1, 2, 3], [3, 5, 6, 7], [8, 9, 10, 11]],
                     [[12, 12, 14, 15], [16, 16, 18, 19], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], 0:2, [0, 1]] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), 0:2, Tensor([0, 1])] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                     [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], True, [0, 1]] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), Tensor(True), Tensor([0, 1])] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                     [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], None, [0, 1]] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), None, Tensor([0, 1])] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"

    np_x = np.array([[[-1, 1, 2, 3], [3, 5, 6, 7], [7, 9, 10, 11]],
                     [[12, 12, 14, 15], [16, 16, 18, 19], [20, 20, 22, 23]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[[0, 1], ..., [0, 1]] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4))
    ms_x[Tensor([0, 1]), ..., Tensor([0, 1])] += -1
    assert np.allclose(np_x, ms_x.asnumpy()), f"np_x:{np_x} {np_x.shape}, ms_x:{ms_x} {ms_x.shape}"


class Net_setitem_index_grad1(nn.Cell):
    def construct(self, x, i):
        x = ops.abs(x)
        # basic index
        if i == 0:
            x[0] = -1
        if i == 1:
            x[0:2] = -1
        if i == 2:
            x[True] = -1
        if i == 3:
            x[False] = -1
        if i == 4:
            x[None] = -1
        if i == 5:
            x[...] = -1
        if i == 6:
            x[0, 0:2, True] = -1
        if i == 7:
            x[0:2, None, ...] = -1
        if i == 8:
            x[[0, 1]] = -1
        if i == 9:
            x[[0, 1]] = -1.0
        # tensor index
        if i == 10:
            x[Tensor(0)] = -1
        if i == 11:
            x[Tensor(True)] = -1
        if i == 12:
            x[Tensor(False)] = -1
        if i == 13:
            x[Tensor(0):Tensor(2)] = -1
        if i == 14:
            x[Tensor([0, 1])] = -1
        # tuple index
        if i == 15:
            x[0, 0:2, True] = -1
        if i == 16:
            x[[0, None, ...]] = -1
        # fancy index
        if i == 17:
            x[[0, 1], [0, 1]] = -1
        if i == 18:
            x[Tensor([0, 1]), Tensor([0, 1])] = -1
        if i == 19:
            x[[0, 1], 0, [0, 1]] = -1
        if i == 20:
            x[Tensor([0, 1]), Tensor(0), Tensor([0, 1])] = -1
        if i == 21:
            x[0, [0, 1], [0, 1]] = -1
        if i == 22:
            x[Tensor(0), Tensor([0, 1]), Tensor([0, 1])] = -1
        if i == 23:
            x[[0, 1], 0:2, [0, 1]] = -1
        if i == 24:
            x[Tensor([0, 1]), 0:2, Tensor([0, 1])] = -1
        if i == 25:
            x[[0, 1], True, [0, 1]] = -1
        if i == 26:
            x[Tensor([0, 1]), Tensor(True), Tensor([0, 1])] = -1
        if i == 27:
            x[[0, 1], None, [0, 1]] = -1
        if i == 28:
            x[Tensor([0, 1]), None, Tensor([0, 1])] = -1
        if i == 29:
            x[[0, 1], ..., [0, 1]] = -1
        if i == 30:
            x[Tensor([0, 1]), ..., Tensor([0, 1])] = -1
        return x


pt_y0 = [[[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]],
         [[1., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]]]
pt_y0_np = np.array(pt_y0)

pt_y1 = [[[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]],
         [[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]]]
pt_y1_np = np.array(pt_y1)

pt_y2 = [[[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]],
         [[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]]]
pt_y2_np = np.array(pt_y2)

pt_y3 = [[[0., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]],
         [[1., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]]]
pt_y3_np = np.array(pt_y3)

pt_y4 = [[[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]],
         [[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]]]
pt_y4_np = np.array(pt_y4)

pt_y5 = [[[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]],
         [[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]]]
pt_y5_np = np.array(pt_y5)

pt_y6 = [[[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [1., 1., 1., 1.,]],
         [[1., 1., 1., 1.,],
          [1., 1., 1., 1.,],
          [1., 1., 1., 1.,]]]
pt_y6_np = np.array(pt_y6)

pt_y7 = [[[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]],
         [[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]]]
pt_y7_np = np.array(pt_y7)

pt_y8 = [[[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]],
         [[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]]]
pt_y8_np = np.array(pt_y8)

pt_y9 = [[[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]],
         [[0., 0., 0., 0.,],
          [0., 0., 0., 0.,],
          [0., 0., 0., 0.,]]]
pt_y9_np = np.array(pt_y9)

pt_y10 = [[[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y10_np = np.array(pt_y10)

pt_y11 = [[[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y11_np = np.array(pt_y11)

pt_y12 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y12_np = np.array(pt_y12)

pt_y13 = [[[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y13_np = np.array(pt_y13)

pt_y14 = [[[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]]]
pt_y14_np = np.array(pt_y14)

pt_y15 = [[[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y15_np = np.array(pt_y15)

pt_y16 = [[[0., 0., 0., 0.,],
           [0., 0., 0., 0.,],
           [0., 0., 0., 0.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y16_np = np.array(pt_y16)

pt_y17 = [[[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [1., 1., 1., 1.,]]]
pt_y17_np = np.array(pt_y17)

pt_y18 = [[[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [1., 1., 1., 1.,]]]
pt_y18_np = np.array(pt_y18)

pt_y19 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 0., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y19_np = np.array(pt_y19)

pt_y20 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 0., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y20_np = np.array(pt_y20)

pt_y21 = [[[0., 1., 1., 1.,],
           [1., 0., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y21_np = np.array(pt_y21)

pt_y22 = [[[0., 1., 1., 1.,],
           [1., 0., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y22_np = np.array(pt_y22)

pt_y23 = [[[0., 1., 1., 1.,],
           [0., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 0., 1., 1.,],
           [1., 0., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y23_np = np.array(pt_y23)

pt_y24 = [[[0., 1., 1., 1.,],
           [0., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 0., 1., 1.,],
           [1., 0., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_y24_np = np.array(pt_y24)

pt_y25 = [[[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [1., 1., 1., 1.,]]]
pt_y25_np = np.array(pt_y25)

pt_y26 = [[[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [1., 1., 1., 1.,]]]
pt_y26_np = np.array(pt_y26)

pt_y27 = [[[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [1., 1., 1., 1.,]]]
pt_y27_np = np.array(pt_y27)

pt_y28 = [[[0., 0., 0., 0.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [0., 0., 0., 0.,],
           [1., 1., 1., 1.,]]]
pt_y28_np = np.array(pt_y28)

pt_y29 = [[[0., 1., 1., 1.,],
           [0., 1., 1., 1.,],
           [0., 1., 1., 1.,]],
          [[1., 0., 1., 1.,],
           [1., 0., 1., 1.,],
           [1., 0., 1., 1.,]]]
pt_y29_np = np.array(pt_y29)

pt_y30 = [[[0., 1., 1., 1.,],
           [0., 1., 1., 1.,],
           [0., 1., 1., 1.,]],
          [[1., 0., 1., 1.,],
           [1., 0., 1., 1.,],
           [1., 0., 1., 1.,]]]
pt_y30_np = np.array(pt_y30)

pt_result1 = [pt_y0_np, pt_y1_np, pt_y2_np, pt_y3_np, pt_y4_np, pt_y5_np, pt_y6_np, pt_y7_np, pt_y8_np, pt_y9_np,
              pt_y10_np, pt_y11_np, pt_y12_np, pt_y13_np, pt_y14_np, pt_y15_np, pt_y16_np, pt_y17_np, pt_y18_np,
              pt_y19_np, pt_y20_np, pt_y21_np, pt_y22_np, pt_y23_np, pt_y24_np, pt_y25_np, pt_y26_np, pt_y27_np,
              pt_y28_np, pt_y29_np, pt_y30_np]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE,])
def test_setitem_refactor_grad1(mode):
    """
    Feature: tensor setitem grad
    Description: Verify the result of tensor setitem grad1 with refactor cpp codes
    Expectation: success
    """
    ms.set_context(mode=mode)

    k = 31
    for i in range(k):
        x_np = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)

        net = Net_setitem_index_grad1()
        net.set_inputs()
        ms_grad_func = ms.grad(net)

        ms_x = Tensor(x_np)
        ms_grad = ms_grad_func(ms_x, i)

        assert np.allclose(pt_result1[i], ms_grad.asnumpy(
            )), f"i:{i}, pt_x.grad:{pt_result1[i]} {pt_result1[i].shape}, ms_grad:{ms_grad} {ms_grad.shape}"



class Net_setitem_index_grad2(nn.Cell):
    def construct(self, x, i):
        x = ops.abs(x)
        # basic index
        if i == 0:
            x[0] += -1
        if i == 1:
            x[0:2] += -1
        if i == 2:
            x[True] += -1
        if i == 3:
            x[False] += -1
        if i == 4:
            x[None] += -1
        if i == 5:
            x[...] += -1
        if i == 6:
            x[0, 0:2, True] += -1
        if i == 7:
            x[0:2, None, ...] += -1
        if i == 8:
            x[[0, 1]] += -1
        if i == 9:
            x[[0, 1]] += -1.0
        # tensor index
        if i == 10:
            x[Tensor(0)] += -1
        if i == 11:
            x[Tensor(True)] += -1
        if i == 12:
            x[Tensor(False)] += -1
        if i == 13:
            x[Tensor(0):Tensor(2)] += -1
        if i == 14:
            x[Tensor([0, 1])] += -1
        # tuple index
        if i == 15:
            x[0, 0:2, True] += -1
        if i == 16:
            x[[0, None, ...]] += -1
        # fancy index
        if i == 17:
            x[[0, 1], [0, 1]] += -1
        if i == 18:
            x[Tensor([0, 1]), Tensor([0, 1])] += -1
        if i == 19:
            x[[0, 1], 0, [0, 1]] += -1
        if i == 20:
            x[Tensor([0, 1]), Tensor(0), Tensor([0, 1])] += -1
        if i == 21:
            x[0, [0, 1], [0, 1]] += -1
        if i == 22:
            x[Tensor(0), Tensor([0, 1]), Tensor([0, 1])] += -1
        if i == 23:
            x[[0, 1], 0:2, [0, 1]] += -1
        if i == 24:
            x[Tensor([0, 1]), 0:2, Tensor([0, 1])] += -1
        if i == 25:
            x[[0, 1], True, [0, 1]] += -1
        if i == 26:
            x[Tensor([0, 1]), Tensor(True), Tensor([0, 1])] += -1
        if i == 27:
            x[[0, 1], None, [0, 1]] += -1
        if i == 28:
            x[Tensor([0, 1]), None, Tensor([0, 1])] += -1
        if i == 29:
            x[[0, 1], ..., [0, 1]] += -1
        if i == 30:
            x[Tensor([0, 1]), ..., Tensor([0, 1])] += -1
        if i == 31:
            x[True] *= 3
        if i == 32:
            x[None] *= 3
        return x


pt_yy0 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_yy0_np = np.array(pt_yy0)

pt_yy1 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_yy1_np = np.array(pt_yy1)

pt_yy2 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_yy2_np = np.array(pt_yy2)

pt_yy3 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_yy3_np = np.array(pt_yy3)

pt_yy4 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_yy4_np = np.array(pt_yy4)

pt_yy5 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_yy5_np = np.array(pt_yy5)

pt_yy6 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_yy6_np = np.array(pt_yy6)

pt_yy7 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_yy7_np = np.array(pt_yy7)

pt_yy8 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_yy8_np = np.array(pt_yy8)

pt_yy9 = [[[0., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]],
          [[1., 1., 1., 1.,],
           [1., 1., 1., 1.,],
           [1., 1., 1., 1.,]]]
pt_yy9_np = np.array(pt_yy9)

pt_yy10 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy10_np = np.array(pt_yy10)

pt_yy11 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy11_np = np.array(pt_yy11)

pt_yy12 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy12_np = np.array(pt_yy12)

pt_yy13 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy13_np = np.array(pt_yy13)

pt_yy14 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy14_np = np.array(pt_yy14)

pt_yy15 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy15_np = np.array(pt_yy15)

pt_yy16 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy16_np = np.array(pt_yy16)

pt_yy17 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy17_np = np.array(pt_yy17)

pt_yy18 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy18_np = np.array(pt_yy18)

pt_yy19 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy19_np = np.array(pt_yy19)

pt_yy20 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy20_np = np.array(pt_yy20)

pt_yy21 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy21_np = np.array(pt_yy21)

pt_yy22 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy22_np = np.array(pt_yy22)

pt_yy23 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy23_np = np.array(pt_yy23)

pt_yy24 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy24_np = np.array(pt_yy24)

pt_yy25 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy25_np = np.array(pt_yy25)

pt_yy26 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy26_np = np.array(pt_yy26)

pt_yy27 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy27_np = np.array(pt_yy27)

pt_yy28 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy28_np = np.array(pt_yy28)

pt_yy29 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy29_np = np.array(pt_yy29)

pt_yy30 = [[[0., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]],
           [[1., 1., 1., 1.,],
            [1., 1., 1., 1.,],
            [1., 1., 1., 1.,]]]
pt_yy30_np = np.array(pt_yy30)

pt_yy31 = [[[0., 3., 3., 3.,],
            [3., 3., 3., 3.,],
            [3., 3., 3., 3.,]],
           [[3., 3., 3., 3.,],
            [3., 3., 3., 3.,],
            [3., 3., 3., 3.,]]]
pt_yy31_np = np.array(pt_yy31)

pt_yy32 = [[[0., 3., 3., 3.,],
            [3., 3., 3., 3.,],
            [3., 3., 3., 3.,]],
           [[3., 3., 3., 3.,],
            [3., 3., 3., 3.,],
            [3., 3., 3., 3.,]]]
pt_yy32_np = np.array(pt_yy32)

pt_result2 = [pt_yy0_np, pt_yy1_np, pt_yy2_np, pt_yy3_np, pt_yy4_np, pt_yy5_np, pt_yy6_np, pt_yy7_np,
              pt_yy8_np, pt_yy9_np, pt_yy10_np, pt_yy11_np, pt_yy12_np, pt_yy13_np, pt_yy14_np, pt_yy15_np,
              pt_yy16_np, pt_yy17_np, pt_yy18_np, pt_yy19_np, pt_yy20_np, pt_yy21_np, pt_yy22_np, pt_yy23_np,
              pt_yy24_np, pt_yy25_np, pt_yy26_np, pt_yy27_np, pt_yy28_np, pt_yy29_np, pt_yy30_np, pt_yy31_np,
              pt_yy32_np]


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE,])
def test_setitem_refactor_grad2(mode):
    """
    Feature: tensor setitem grad
    Description: Verify the result of tensor setitem grad2 with refactor cpp codes
    Expectation: success
    """
    ms.set_context(mode=mode)

    k = len(pt_result2)
    for i in range(k):
        x_np = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)

        net = Net_setitem_index_grad2()
        net.set_inputs()
        ms_grad_func = ms.grad(net)

        ms_x = Tensor(x_np)
        ms_grad = ms_grad_func(ms_x, i)

        assert np.allclose(pt_result2[i], ms_grad.asnumpy(
            )), f"i:{i}, pt_x.grad:{pt_result2[i]} {pt_result2[i].shape}, ms_grad:{ms_grad} {ms_grad.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE,])
def test_setitem_refactor_exception(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem exception with refactor codes
    Expectation: success
    """
    ms.set_context(mode=mode)

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    with pytest.raises(IndexError) as exc:
        ms_x[2, 0, 0] = -1
    assert "is out of bounds for dimension" in str(exc.value)

    with pytest.raises(IndexError) as exc:
        ms_x[0, 0, 0, 0] = -1
    assert "too many indices for tensor with dimension size" in str(exc.value)

    with pytest.raises(IndexError) as exc:
        ms_x[0, 't'] = -1
    assert "Invalid tensor index type" in str(exc.value)

    with pytest.raises(ValueError) as exc:
        ms_x[0:3:-1] = -1
    assert "slice step must be positive" in str(exc.value)

    with pytest.raises(TypeError) as exc:
        ms_x[0] = (1, 2, 3)
    assert "Can't assign a <class 'tuple'> to a Float32" in str(exc.value)

    with pytest.raises(TypeError) as exc:
        ms_x[0] = [1, 2, 3]
    assert "Can't assign a <class 'list'> to a Float32" in str(exc.value)

    ms_x = Tensor(0)
    with pytest.raises(TypeError) as exc:
        ms_x[0] = -1
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    with pytest.raises(TypeError) as exc:
        ms_x[0:1] = -1
    assert "Invalid index of a 0-dim tensor." in str(exc.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE,])
def test_setitem_graph_mode(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem in graph mode
    Expectation: success
    """
    ms.set_context(mode=mode)
    np_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    ms_x = Tensor(np_x)
    ms_x[0] = -1
    np_expect = np.array([[-1, -1, -1], [3, 4, 5]]).astype(np.float32)
    assert np.allclose(np_expect, ms_x.asnumpy()), f"np_expect:{np_expect}, ms_x:{ms_x}"
