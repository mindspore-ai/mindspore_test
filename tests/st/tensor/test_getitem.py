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

import os
import numpy as np
import pytest
from tests.mark_utils import arg_mark
from tests.st.pi_jit.share.utils import get_code_extra, has_graph

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops, context


def assert_executed_by_graph_mode(func, x, index):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    if jcr is not None:
        assert jcr['stat'] == 'GRAPH_CALLABLE', f"ms_x: {x}, index: {index}"
        assert jcr['break_count_'] == 0, f'break_count expect: 0, actual: {jcr["break_count_"]}, '\
                                         f"ms_x: {x}, index: {index}"
        assert has_graph(jcr), f"ms_x: {x}, index: {index}"


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
    context.set_context(jit_level='O0')
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
    context.set_context(jit_level='O0')
    net = Net_index2_slice()
    x_np = np.arange(6 * 7 * 8 * 9).reshape((6, 7, 8, 9)).astype(np.float32)
    index1_np = -1
    index2_np = -1
    y_np = x_np[index1_np, 0:2, index2_np]
    y = net(Tensor(x_np), Tensor(index1_np), Tensor(index2_np))
    assert np.allclose(y_np, y.asnumpy())


def getitem_check_indexing(x, index, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:

        def func(ms_x, index):
            ms_y = ms_x[index]
            return ms_y
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def func(ms_x, index):
            ms_y = ms_x[index]
            return ms_y

    ms_output = func(x, index)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(func, x, index)

    assert np.allclose(np_expected, ms_output.asnumpy()), f"ms_x: {x}, index: {index}, " \
                                                          f"expected:{np_expected} {np_expected.shape}, " \
                                                          f"ms_output:{ms_output} {ms_output.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast', 'bytecode'])
def test_getitem(capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)

    # Basic index
    basic_indices = [0, slice(0, 1), True, False, None, ..., (0, 2, ...), [0, 1]]
    for index in basic_indices:
        np_expected = np_x[index]
        getitem_check_indexing(ms_x, index, np_expected, capture_mode)

    # Numpy index
    if capture_mode is None:
        numpy_indices = [np.array(0), np.array(True), np.array(False)]
        for index in numpy_indices:
            np_expected = np_x[index]
            getitem_check_indexing(ms_x, index, np_expected, capture_mode)

    # Tensor index
    tensor_indices = [Tensor(0), Tensor(True), Tensor(False), slice(Tensor(0), Tensor(2)), Tensor([0, 1])]
    np_indices = [0, True, False, slice(0, 2), [0, 1]]
    for np_index, tensor_index in zip(np_indices, tensor_indices):
        np_expected = np_x[np_index]
        getitem_check_indexing(ms_x, tensor_index, np_expected, capture_mode)

    # Tuple index
    tuple_indices = [(0, slice(0, 2), True), (0, None, ...)]
    np_expecteds = [
        np.array([[[0, 1, 2, 3]], [[4, 5, 6, 7]]]),
        np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]])
    ]
    for index, np_expected in zip(tuple_indices, np_expecteds):
        getitem_check_indexing(ms_x, index, np_expected, capture_mode)

    # Fancy index
    fancy_indices = [([0, 1], [0, 1]),
                     (Tensor([0, 1]), Tensor([0, 1])),
                     ([0, 1], 0, [0, 1]),
                     (Tensor([0, 1]), Tensor(0), Tensor([0, 1])),
                     (0, [0, 1], [0, 1]),
                     (Tensor(0), Tensor([0, 1]), Tensor([0, 1])),
                     ([0, 1], slice(0, 2), [0, 1]),
                     (Tensor([0, 1]), slice(0, 2), Tensor([0, 1])),
                     ([0, 1], True, [0, 1]),
                     (Tensor([0, 1]), Tensor(True), Tensor([0, 1])),
                     ([0, 1], None, [0, 1]),
                     (Tensor([0, 1]), None, Tensor([0, 1])),
                     ([0, 1], ..., [0, 1]),
                     (Tensor([0, 1]), ..., Tensor([0, 1]))]

    np_expecteds = [np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([0, 13]),
                    np.array([0, 13]),
                    np.array([0, 5]),
                    np.array([0, 5]),
                    np.array([[0, 4], [13, 17]]),
                    np.array([[0, 4], [13, 17]]),
                    np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([[0, 1, 2, 3], [16, 17, 18, 19]]),
                    np.array([[[0, 1, 2, 3]], [[16, 17, 18, 19]]]),
                    np.array([[[0, 1, 2, 3]], [[16, 17, 18, 19]]]),
                    np.array([[0, 4, 8], [13, 17, 21]]),
                    np.array([[0, 4, 8], [13, 17, 21]])]

    for index, np_expected in zip(fancy_indices, np_expecteds):
        getitem_check_indexing(ms_x, index, np_expected, capture_mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast'])
def test_getitem2(capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)

    fancy_indices = [([Tensor(0), 1], Tensor(0), [0, Tensor(1)]),
                     (Tensor(0), [0, Tensor(1)], [Tensor(0), Tensor(1)])]

    np_expecteds = [np.array([0, 13]),
                    np.array([0, 5])]

    for index, np_expected in zip(fancy_indices, np_expecteds):
        getitem_check_indexing(ms_x, index, np_expected, capture_mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('capture_mode', [None, 'ast', 'bytecode'])
def test_getitem_with_ellipsis(capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem with ellipsis
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_x = np.arange(2 * 3 * 4 * 5).reshape((2, 3, 4, 5))
    ms_x = Tensor(np_x)

    ellipsis_indices = [(1, ..., True, [0, 1]),
                        (Tensor(1), ..., Tensor(True), Tensor([0, 1])),
                        (slice(0, 1), ..., None)]
    np_expecteds = [
        np.array([[[60, 61], [65, 66], [70, 71], [75, 76]], [[80, 81], [85, 86], [90, 91], [95, 96]],
                  [[100, 101], [105, 106], [110, 111], [115, 116]]]),
        np.array([[[60, 61], [65, 66], [70, 71], [75, 76]], [[80, 81], [85, 86], [90, 91], [95, 96]],
                  [[100, 101], [105, 106], [110, 111], [115, 116]]]),
        np.array([[[[[0], [1], [2], [3], [4]], [[5], [6], [7], [8], [9]], [[10], [11], [12], [13], [14]],
                    [[15], [16], [17], [18], [19]]],
                   [[[20], [21], [22], [23], [24]], [[25], [26], [27], [28], [29]], [[30], [31], [32], [33], [34]],
                    [[35], [36], [37], [38], [39]]],
                   [[[40], [41], [42], [43], [44]], [[45], [46], [47], [48], [49]], [[50], [51], [52], [53], [54]],
                    [[55], [56], [57], [58], [59]]]]])
    ]

    for index, np_expected in zip(ellipsis_indices, np_expecteds):
        getitem_check_indexing(ms_x, index, np_expected, capture_mode)


class NetGetitem(nn.Cell):

    def __init__(self, index):
        super().__init__()
        self.index = index

    def construct(self, x):
        x = ops.relu(x)
        y = x[self.index]
        return y


def getitem_check_grad(x, index, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def grad_func(net, x):
            return ms.grad(net)(x)
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def grad_func(net, x):
            return ms.grad(net)(x)

    net = NetGetitem(index)
    ms_grad = grad_func(net, x)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(net.construct, x, index)

    assert np.allclose(np_expected, ms_grad.asnumpy()), f"ms_x: {x}, index: {index}, " \
                                                        f"expected:{np_expected} {np_expected.shape}, " \
                                                        f"ms_grad:{ms_grad} {ms_grad.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast', 'bytecode'])
def test_getitem_grad_index_negative(capture_mode):
    """
    Feature: tensor getitem grad
    Description: Verify the result of tensor getitem grad with negative index
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    index = (Tensor(-1), Tensor(-1), Tensor(-1))
    np_expected = np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                            [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 1.,]]])
    getitem_check_grad(ms_x, index, np_expected, capture_mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast', 'bytecode'])
def test_getitem_grad(capture_mode):
    """
    Feature: tensor getitem grad
    Description: Verify the result of tensor getitem grad
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    # Base index
    base_indices = [0, slice(0, 2), True, False, ..., None, [0, 1]]
    np_expecteds = [np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]])]
    for index, np_expected in zip(base_indices, np_expecteds):
        ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
        getitem_check_grad(ms_x, index, np_expected, capture_mode)

    # Tensor index
    tensor_indices = [Tensor(0), Tensor(True), Tensor(False), slice(Tensor(0), Tensor(2)), Tensor([0, 1])]
    np_expecteds = [np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]])]
    for index, np_expected in zip(tensor_indices, np_expecteds):
        getitem_check_grad(ms_x, index, np_expected, capture_mode)

    # Tuple index
    tuple_indices = [(0, slice(0, 2), True), (0, None, ...)]
    np_expecteds = [np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]])]
    for index, np_expected in zip(tuple_indices, np_expecteds):
        getitem_check_grad(ms_x, index, np_expected, capture_mode)

    # Fancy index
    fancy_indices = [([0, 1], [0, 1]),
                     (Tensor([0, 1]), Tensor([0, 1])),
                     ([0, 1], 0, [0, 1]),
                     (Tensor([0, 1]), Tensor(0), Tensor([0, 1])),
                     (0, [0, 1], [0, 1]),
                     (Tensor(0), Tensor([0, 1]), Tensor([0, 1])),
                     ([0, 1], slice(0, 2), [0, 1]),
                     (Tensor([0, 1]), slice(0, 2), Tensor([0, 1])),
                     ([0, 1], True, [0, 1]),
                     (Tensor([0, 1]), Tensor(True), Tensor([0, 1])),
                     ([0, 1], None, [0, 1]),
                     (Tensor([0, 1]), None, Tensor([0, 1])),
                     ([0, 1], ..., [0, 1]),
                     (Tensor([0, 1]), ..., Tensor([0, 1]))]
    np_expecteds = [np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 1., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 1., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [1., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 1., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [1., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 1., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [1., 1., 1., 1.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [1., 0., 0., 0.,], [1., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 1., 0., 0.,], [0., 1., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [1., 0., 0., 0.,], [1., 0., 0., 0.,]],
                              [[0., 1., 0., 0.,], [0., 1., 0., 0.,], [0., 1., 0., 0.,]]])]
    for index, np_expected in zip(fancy_indices, np_expecteds):
        getitem_check_grad(ms_x, index, np_expected, capture_mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'ast'), (ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_getitem_exception(mode, capture_mode):
    """
    Feature: tensor getitem
    Description: Verify the result of tensor getitem exception
    Expectation: success
    """
    ms.set_context(jit_config={"jit_level": "O0"})
    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func1(x): return ms_x[2, 0, 0]
    with pytest.raises(IndexError) as exc:
        _ = ms_x[2, 0, 0] if mode == ms.PYNATIVE_MODE else func1(ms_x)
    assert "is out of bounds" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func2(x): return ms_x[0, 0, 0, 0]
    with pytest.raises(IndexError) as exc:
        _ = ms_x[0, 0, 0, 0] if mode == ms.PYNATIVE_MODE else func2(ms_x)
    assert "too many indices" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func3(x): return ms_x[0, 't']
    with pytest.raises(IndexError) as exc:
        _ = ms_x[0, 't'] if mode == ms.PYNATIVE_MODE else func3(ms_x)
    assert "Invalid tensor index type" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func4(x): return ms_x[0:3:-1]
    with pytest.raises(ValueError) as exc:
        _ = ms_x[0:3:-1] if mode == ms.PYNATIVE_MODE else func4(ms_x)
    assert "slice step must be positive" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func5(x): return ms_x[0]
    ms_x = Tensor(0)
    with pytest.raises(TypeError) as exc:
        _ = ms_x[0] if mode == ms.PYNATIVE_MODE else func5(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func6(x): return ms_x[0:1]
    with pytest.raises(TypeError) as exc:
        _ = ms_x[0:1] if mode == ms.PYNATIVE_MODE else func6(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    with pytest.raises(TypeError) as exc:
        _ = sum(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)
