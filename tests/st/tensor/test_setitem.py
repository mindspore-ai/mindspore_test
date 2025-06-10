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
from mindspore import Tensor, ops


def assert_executed_by_graph_mode(func, x, index, value):
    jcr = get_code_extra(getattr(func, "__wrapped__", func))
    if jcr is not None:
        assert jcr['stat'] == 'GRAPH_CALLABLE', f"ms_x: {x}, index: {index}, value: {value}"
        assert jcr['break_count_'] == 0, f'break_count expect: 0, actual: {jcr["break_count_"]}, '\
                                         f"ms_x: {x}, index: {index}, value: {value}"
        assert has_graph(jcr), f"ms_x: {x}, index: {index}, value: {value}"


def is_index_need_skip(index, skip_list):
    """cheeck if index need skip, used for debug"""
    def check_index_same(index, to_skip):
        if type(to_skip) != type(index):  # pylint: disable=unidiomatic-typecheck
            return False
        if isinstance(to_skip, Tensor):
            result = (index == to_skip).all()
        elif isinstance(to_skip, (tuple, list)):
            result = all(check_index_same(index[i], to_skip[i]) for i in range(len(to_skip)))
        else:
            result = index == to_skip
        return bool(result)

    for to_skip in skip_list:
        if check_index_same(index, to_skip):
            return True

    return False


def setitem_check_indexing(x, index, value, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def func(ms_x, index, value):
            ms_x[index] = value
            return ms_x
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def func(ms_x, index, value):
            ms_x[index] = value
            return ms_x

    ms_output = func(x, index, value)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(func, x, index, value)

    assert np.allclose(np_expected, ms_output.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                          f"expected:{np_expected} {np_expected.shape}, " \
                                                          f"ms_output:{ms_output} {ms_output.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast'])
def test_setitem(capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    # Basic index
    basic_indices = [0, slice(0, 1), True, False, None, ..., (0, 2, ...), [0, 1]]
    for index in basic_indices:
        np_x = np.arange(2*3*4).reshape(2, 3, 4)
        ms_x = Tensor(np_x)
        value = -1
        np_x[index] = value
        setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Basic index with float value
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    index = [0, 1]
    value = -1.0
    np_x[index] = value
    setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Basic index with tensor value
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    index = 0
    value = Tensor([[[-1, -1, -1, -1]]])
    np_expected = np.array([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    setitem_check_indexing(ms_x, index, value, np_expected, capture_mode)

    # Numpy index
    if capture_mode is None:
        numpy_indices = [np.array(0), np.array(True), np.array(False), (np.array(0), np.array(1)), np.array([0, 1])]
        for index in numpy_indices:
            np_x = np.arange(2*3*4).reshape(2, 3, 4)
            ms_x = Tensor(np_x)
            value = -1
            np_x[index] = value
            setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Tensor index
    tensor_indices = [Tensor(0), Tensor(True), Tensor(False), slice(Tensor(0), Tensor(2)), Tensor([0, 1])]
    np_indices = [0, True, False, slice(0, 2), [0, 1]]
    for index, np_index in zip(tensor_indices, np_indices):
        np_x = np.arange(2*3*4).reshape(2, 3, 4)
        ms_x = Tensor(np_x)
        value = -1
        np_x[np_index] = value
        setitem_check_indexing(ms_x, index, value, np_x, capture_mode)

    # Tuple index
    tuple_index = (0, None, ...)
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    value = -1
    np_expected = np.array([[[-1, -1, -1, -1], [-1, -1, -1, -1], [-1, -1, -1, -1]],
                            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    setitem_check_indexing(ms_x, tuple_index, value, np_expected, capture_mode)

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
                     (Tensor([0, 1]), ..., Tensor([0, 1])),
                     (Tensor([0]), Tensor(0), slice(0, 4, 2))]
    np_expecteds = [np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, -1, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, -1, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [-1, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, -1, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [-1, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, -1, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, -1, -1, -1], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [-1, -1, -1, -1], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [-1, 5, 6, 7], [-1, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, -1, 18, 19], [20, -1, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [-1, 5, 6, 7], [-1, 9, 10, 11]],
                              [[12, -1, 14, 15], [16, -1, 18, 19], [20, -1, 22, 23]]]),
                    np.array([[[-1, 1, -1, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])]
    for index, np_expected in zip(fancy_indices, np_expecteds):
        ms_x = Tensor(np.arange(2*3*4).reshape(2, 3, 4))
        value = -1
        setitem_check_indexing(ms_x, index, value, np_expected, capture_mode)


def setitem_check_iadd_indexing(x, index, value, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def func(ms_x, index, value):
            ms_x[index] += value
            return ms_x
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def func(ms_x, index, value):
            ms_x[index] += value
            return ms_x

    ms_output = func(x, index, value)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(func, x, index, value)

    assert np.allclose(np_expected, ms_output.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                          f"expected:{np_expected} {np_expected.shape}, " \
                                                          f"ms_output:{ms_output} {ms_output.shape}"

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast'])
def test_setitem_with_iadd(capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem with iadd
    Expectation: success
    """

    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    # Basic index
    basic_indices = [0, slice(0, 1), True, False, None, ..., (0, slice(0, 2), True), (slice(0, 2), None, ...), [0, 1]]
    for index in basic_indices:
        np_x = np.arange(2*3*4).reshape(2, 3, 4)
        ms_x = Tensor(np_x)
        value = -1
        np_x[index] += value
        setitem_check_iadd_indexing(ms_x, index, value, np_x, capture_mode)

    # Tuple index
    tuple_index = (0, None, ...)
    np_x = np.arange(2*3*4).reshape(2, 3, 4)
    ms_x = Tensor(np_x)
    value = -1
    np_expected = np.array([[[-1, 0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10]],
                            [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]])
    setitem_check_iadd_indexing(ms_x, tuple_index, value, np_expected, capture_mode)

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
    np_expecteds = [np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 4, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [4, 4, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [16, 17, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [3, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 16, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [3, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 16, 18, 19], [20, 21, 22, 23]]]),
                    np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 0, 1, 2], [4, 5, 6, 7], [8, 9, 10, 11]],
                              [[12, 13, 14, 15], [15, 16, 17, 18], [20, 21, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [3, 5, 6, 7], [7, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 16, 18, 19], [20, 20, 22, 23]]]),
                    np.array([[[-1, 1, 2, 3], [3, 5, 6, 7], [7, 9, 10, 11]],
                              [[12, 12, 14, 15], [16, 16, 18, 19], [20, 20, 22, 23]]])]
    for index, np_expected in zip(fancy_indices, np_expecteds):
        ms_x = Tensor(np.arange(2*3*4).reshape(2, 3, 4))
        value = -1
        setitem_check_iadd_indexing(ms_x, index, value, np_expected, capture_mode)


class NetSetitem(nn.Cell):

    def __init__(self, index, value):
        super().__init__()
        self.index = index
        self.value = value

    def construct(self, x):
        x = ops.abs(x)
        x[self.index] = self.value
        return x


def setitem_check_grad(x, index, value, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def grad_func(net, x):
            return ms.grad(net)(x)
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def grad_func(net, x):
            return ms.grad(net)(x)

    net = NetSetitem(index, value)
    ms_grad = grad_func(net, x)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(net.construct, x, index, value)

    assert np.allclose(np_expected, ms_grad.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                        f"expected:{np_expected} {np_expected.shape}, " \
                                                        f"ms_grad:{ms_grad} {ms_grad.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast', 'bytecode'])
def test_setitem_grad(capture_mode):
    """
    Feature: tensor setitem grad
    Description: Verify the result of tensor setitem grad1
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    # Base index
    base_indices = [0, slice(0, 2), True, False, ..., None, [0, 1]]
    np_expecteds = [np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]])]
    for index, np_expected in zip(base_indices, np_expecteds):
        ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
        value = -1
        setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

    # Basic index with float value
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    index = [0, 1]
    value = -1.0
    np_expected = np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                            [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]])
    setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

    # Basic index with tensor value
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    index = 0
    value = Tensor([[[-1, -1, -1, -1]]])
    np_expected = np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                            [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]])
    setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

    # Tensor index
    tensor_indices = [Tensor(0), Tensor(True), Tensor(False), slice(Tensor(0), Tensor(2)), Tensor([0, 1])]
    value = -1
    np_expecteds = [np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]]])]
    for index, np_expected in zip(tensor_indices, np_expecteds):
        setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

    # Tuple index
    tuple_indices = [(0, slice(0, 2), True), (0, None, ...)]
    value = -1
    np_expecteds = [np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [1., 1., 1., 1.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]]),
                    np.array([[[0., 0., 0., 0.,], [0., 0., 0., 0.,], [0., 0., 0., 0.,]],
                              [[1., 1., 1., 1.,], [1., 1., 1., 1.,], [1., 1., 1., 1.,]]])]
    for index, np_expected in zip(tuple_indices, np_expecteds):
        setitem_check_grad(ms_x, index, value, np_expected, capture_mode)

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
                     (Tensor([0, 1]), ..., Tensor([0, 1])),
                     (Tensor([0]), Tensor(0), slice(0, 4, 2))]
    value = -1
    np_expecteds = [
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [1., 0., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [1., 0., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [0., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 0., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [0., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 0., 1., 1.], [1., 1., 1., 1.]]]),
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 0., 0., 0.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [0., 0., 0., 0.], [1., 1., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [0., 1., 1., 1.], [0., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 0., 1., 1.], [1., 0., 1., 1.]]]),
        np.array([[[0., 1., 1., 1.], [0., 1., 1., 1.], [0., 1., 1., 1.]],
                  [[1., 0., 1., 1.], [1., 0., 1., 1.], [1., 0., 1., 1.]]]),
        np.array([[[0., 1., 0., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                  [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]])]
    for index, np_expected in zip(fancy_indices, np_expecteds):
        setitem_check_grad(ms_x, index, value, np_expected, capture_mode)


class NetSetitemIadd(nn.Cell):

    def __init__(self, index, value):
        super().__init__()
        self.index = index
        self.value = value

    def construct(self, x):
        x = ops.abs(x)
        x[self.index] += self.value
        return x


def setitem_check_iadd_grad(x, index, value, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def grad_func(net, x):
            return ms.grad(net)(x)
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def grad_func(net, x):
            return ms.grad(net)(x)

    net = NetSetitemIadd(index, value)
    ms_grad = grad_func(net, x)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(net.construct, x, index, value)

    assert np.allclose(np_expected, ms_grad.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                        f"expected:{np_expected} {np_expected.shape}, " \
                                                        f"ms_grad:{ms_grad} {ms_grad.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast', 'bytecode'])
@pytest.mark.parametrize(
    'index',
    [
        # 0-3: base index
        slice(0, 2),
        False,
        ...,
        [0, 1],
        # 4-5: tensor index
        slice(Tensor(0), Tensor(2)),
        Tensor([0, 1]),
        # 6-9: fancy index
        ([0, 1], [0, 1]),
        (Tensor([0, 1]), Tensor([0, 1])),
        ([0, 1], ..., [0, 1]),
        (Tensor([0, 1]), ..., Tensor([0, 1])),
    ],
)
def test_setitem_grad_with_iadd(capture_mode, index):
    """
    Feature: tensor setitem grad
    Description: Verify the result of tensor setitem grad with iadd
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_expected = np.array([[[0., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                            [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    value = -1
    setitem_check_iadd_grad(ms_x, index, value, np_expected, capture_mode)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast',
                                          pytest.param('bytecode', marks=pytest.mark.skip(reason="Unsupported now"))])
@pytest.mark.parametrize(
    'index',
    [
        # 0-2: base index
        0,
        True,
        None,
        # 3-5: tensor index
        Tensor(0),
        Tensor(True),
        Tensor(False),
        # 6-7: tuple index
        (0, None, ...),
        (0, slice(0, 2), True),
        # 8-17: fancy index
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
    ],
)
def test_setitem_grad_with_iadd_jit_invalid(capture_mode, index):
    """
    Feature: tensor setitem grad
    Description: Verify the result of tensor setitem grad with iadd
    Expectation: Raise exception
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'

    np_expected = np.array([[[0., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]],
                            [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 1., 1., 1.]]])
    ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
    value = -1
    if capture_mode is None:
        setitem_check_iadd_grad(ms_x, index, value, np_expected, capture_mode)
    else:
        with pytest.raises(RuntimeError) as err:
            setitem_check_iadd_grad(ms_x, index, value, np_expected, capture_mode)
        assert "When performing an in-place operation on an object generated by a view operation, " \
                "it is currently not supported to compute gradients for the " \
                "other inputs of this in-place operator" in str(err.value)


class NetSetitemImul(nn.Cell):

    def __init__(self, index, value):
        super().__init__()
        self.index = index
        self.value = value

    def construct(self, x):
        x = ops.abs(x)
        x[self.index] *= self.value
        return x


def setitem_check_imul_grad(x, index, value, np_expected, capture_mode=None):
    """getitem run and check"""
    if capture_mode is None:
        def grad_func(net, x):
            return ms.grad(net)(x)
    else:
        @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
        def grad_func(net, x):
            return ms.grad(net)(x)

    net = NetSetitemImul(index, value)
    ms_grad = grad_func(net, x)

    if capture_mode == 'bytecode':
        assert_executed_by_graph_mode(net.construct, x, index, value)

    assert np.allclose(np_expected, ms_grad.asnumpy()), f"ms_x: {x}, index: {index}, value: {value}, " \
                                                        f"expected:{np_expected} {np_expected.shape}, " \
                                                        f"ms_grad:{ms_grad} {ms_grad.shape}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', [None, 'ast'])
def test_setitem_grad_with_imul(capture_mode):
    """
    Feature: tensor setitem grad
    Description: Verify the result of tensor setitem grad with imul
    Expectation: success
    """
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    if capture_mode is not None:
        os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    iadd_indices = [True, None]
    np_expected = np.array([[[0., 3., 3., 3.], [3., 3., 3., 3.], [3., 3., 3., 3.]],
                            [[3., 3., 3., 3.], [3., 3., 3., 3.], [3., 3., 3., 3.]]])
    for index in iadd_indices:
        ms_x = Tensor(np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32))
        value = 3
        if capture_mode is None:
            setitem_check_imul_grad(ms_x, index, value, np_expected, capture_mode)
        else:
            with pytest.raises(RuntimeError) as err:
                setitem_check_imul_grad(ms_x, index, value, np_expected, capture_mode)
            assert "When performing an in-place operation on an object generated by a view operation, " \
                   "it is currently not supported to compute gradients for the " \
                   "other inputs of this in-place operator" in str(err.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode, capture_mode', [(ms.GRAPH_MODE, 'ast'), (ms.GRAPH_MODE, 'bytecode'),
                                                (ms.PYNATIVE_MODE, 'ast')])
def test_setitem_exception(mode, capture_mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem exception
    Expectation: success
    """
    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'

    np_x = np.arange(2 * 3 * 4).reshape((2, 3, 4)).astype(np.float32)
    ms_x = Tensor(np_x)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func1(x): x[2, 0, 0] = -1; return x
    with pytest.raises(IndexError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[2, 0, 0] = -1
        else:
            _ = func1(ms_x)
    assert "is out of bounds" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func2(x): x[0, 0, 0, 0] = -1; return x
    with pytest.raises(IndexError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0, 0, 0, 0] = -1
        else:
            _ = func2(ms_x)
    assert "too many indices" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func3(x): x[0, 't'] = -1; return x
    with pytest.raises(IndexError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0, 't'] = -1
        else:
            _ = func3(ms_x)
    assert "Invalid tensor index type" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func4(x): x[0:3:-1] = -1; return x
    with pytest.raises(ValueError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0:3:-1] = -1
        else:
            _ = func4(ms_x)
    assert "slice step must be positive" in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func5(x): x[0] = (1, 2, 3); return x
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0] = (1, 2, 3)
        else:
            _ = func5(ms_x)
    assert "the type of value can only be bool, int, float or Tensor." in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func6(x): x[0] = [1, 2, 3]; return x
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0] = [1, 2, 3]
        else:
            _ = func6(ms_x)
    assert "the type of value can only be bool, int, float or Tensor." in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func7(x): x[0] = -1; return x
    ms_x = Tensor(0)
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0] = -1
        else:
            _ = func7(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func8(x): ms_x[0:1] = -1; return ms_x
    with pytest.raises(TypeError) as exc:
        if mode == ms.PYNATIVE_MODE:
            ms_x[0:1] = -1
        else:
            _ = func8(ms_x)
    assert "Invalid index of a 0-dim tensor." in str(exc.value)


class IndexDynamicShapeNet(nn.Cell):
    def construct(self, x, index, value):
        x = ops.abs(x)
        x[0:2, index] = value
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', ['ast', 'bytecode'])
@pytest.mark.parametrize('x_shape', [(3, 3, None), (3, 3, 3)])
@pytest.mark.parametrize('index_shape', [(2, None), (2, 2)])
@pytest.mark.parametrize('value_shape', [(None,), (1,)])
def test_setitem_index_dynamic_shape_test(capture_mode, x_shape, index_shape, value_shape):
    """
    Feature: tensor setitem with index dynamic shape
    Description: Verify the result of tensor setitem with index dynamic shape
    Expectation: success
    """
    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func(net, x, index, value):
        return net(x, index, value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def grad_func(net, x, index, value):
        return ms.grad(net)(x, index, value)

    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'

    pt_result = np.array([[[-1., -1., -1.], [-1., -1., -1.], [6., 7., 8.]],
                          [[-1., -1., -1.], [-1., -1., -1.], [15., 16., 17.]],
                          [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]]])
    pt_grad = np.array([[[0., 0., 0.], [0., 0., 0.], [1., 1., 1.]],
                        [[0., 0., 0.], [0., 0., 0.], [1., 1., 1.]],
                        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])

    net = IndexDynamicShapeNet()
    ms_x = Tensor(np.arange(3 * 3 * 3).reshape((3, 3, 3)).astype(np.float32))
    index = Tensor([[0, 1], [0, 1]])
    value = Tensor(-1, ms.float32)
    x_dyn = Tensor(shape=x_shape, dtype=ms.float32) if None in x_shape else ms_x
    index_dyn = Tensor(shape=index_shape, dtype=ms.int64) if None in index_shape else index
    value_dyn = Tensor(shape=value_shape, dtype=ms.float32) if None in value_shape else value
    net.set_inputs(x_dyn, index_dyn, value_dyn)

    ms_result = func(net, ms_x, index, value)
    assert np.allclose(pt_result, ms_result.asnumpy()), f"pt_result: {pt_result}, " \
                                                        f"ms_result: {ms_result.asnumpy()}"

    ms_grad = grad_func(net, ms_x, index, value)
    assert np.allclose(pt_grad, ms_grad.asnumpy()), f"pt_grad: {pt_grad}, " \
                                                    f"ms_grad: {ms_grad.asnumpy()}"


class IndexDynamicRankNet(nn.Cell):
    def construct(self, x, index1, index2, cond, value):
        x = ops.abs(x)
        if cond:
            index = index1
        else:
            index = index2
        x[0:2, index] = value
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', ['ast', 'bytecode'])
def test_setitem_index_dynamic_rank_test(capture_mode):
    """
    Feature: tensor setitem with index dynamic rank
    Description: Verify the result of tensor setitem with index dynamic rank
    Expectation: success
    """
    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func(net, x, index1, index2, cond, value):
        return net(x, index1, index2, cond, value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def grad_func(net, x, index1, index2, cond, value):
        return ms.grad(net)(x, index1, index2, cond, value)

    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'

    pt_result = np.array([[[0., 1., 2.], [-1., -1., -1.], [6., 7., 8.]],
                          [[9., 10., 11.], [-1., -1., -1.], [15., 16., 17.]],
                          [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]]])
    pt_grad = np.array([[[0., 1., 1.], [0., 0., 0.], [1., 1., 1.]],
                        [[1., 1., 1.], [0., 0., 0.], [1., 1., 1.]],
                        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])

    net = IndexDynamicRankNet()
    ms_x = Tensor(np.arange(3 * 3 * 3).reshape((3, 3, 3)).astype(np.float32))
    index1 = Tensor([1])
    index2 = Tensor([[0, 1], [0, 1]])
    cond = Tensor(True)
    value = Tensor(-1, ms.float32)
    ms_result = func(net, ms_x, index1, index2, cond, value)
    assert np.allclose(pt_result, ms_result.asnumpy()), f"pt_result: {pt_result}, " \
                                                        f"ms_result: {ms_result.asnumpy()}"

    ms_grad = grad_func(net, ms_x, index1, index2, cond, value)
    assert np.allclose(pt_grad, ms_grad.asnumpy()), f"pt_grad: {pt_grad}, " \
                                                    f"ms_grad: {ms_grad.asnumpy()}"



class IndexDynamicRank2Net(nn.Cell):
    def construct(self, x, index1, index2, value):
        x = ops.abs(x)
        mask = index1 == index2
        index = index1[mask]
        value = value[mask]
        x[0:1, index] = value
        return x


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('capture_mode', ['ast', 'bytecode'])
def test_setitem_index_dynamic_rank_test2(capture_mode):
    """
    Feature: tensor setitem with index dynamic rank
    Description: Verify the result of tensor setitem with index dynamic rank
    Expectation: success
    """
    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def func(net, x, index1, index2, value):
        return net(x, index1, index2, value)

    @ms.jit(capture_mode=capture_mode, jit_level="O0", backend="ms_backend")
    def grad_func(net, x, index1, index2, value):
        return ms.grad(net)(x, index1, index2, value)

    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'

    pt_result = np.array([[[0., 1., 2.], [3., 4., 5.], [-3., -3., -3.]],
                          [[9., 10., 11.], [12., 13., 14.], [15., 16., 17.]],
                          [[18., 19., 20.], [21., 22., 23.], [24., 25., 26.]]])
    pt_grad = np.array([[[0., 1., 1.], [1., 1., 1.], [0., 0., 0.]],
                        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
                        [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]])

    net = IndexDynamicRank2Net()
    ms_x = Tensor(np.arange(3 * 3 * 3).reshape((3, 3, 3)).astype(np.float32))
    index1 = Tensor([0, 1, 2])
    index2 = Tensor([1, 2, 2])
    value = Tensor([-1, -2, -3], ms.float32)
    ms_result = func(net, ms_x, index1, index2, value)
    assert np.allclose(pt_result, ms_result.asnumpy()), f"pt_result: {pt_result}, " \
                                                        f"ms_result: {ms_result.asnumpy()}"

    ms_grad = grad_func(net, ms_x, index1, index2, value)
    assert np.allclose(pt_grad, ms_grad.asnumpy()), f"pt_grad: {pt_grad}, " \
                                                    f"ms_grad: {ms_grad.asnumpy()}"


class NetWithIndexAndMul(nn.Cell):
    def construct(self, x):
        a = ms.mint.zeros_like(x)
        a[:, 0:1] = 1
        b = x * a
        return b


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_setitem_with_mul(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of network with mul after setitem.
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    os.environ["MS_DEV_TENSOR_INDEX_BOOST"] = '1'
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'

    ms_x = Tensor([[0, 0], [2, 0]])
    net = NetWithIndexAndMul()
    ms_y = net(ms_x)
    np_expect = np.array([[0, 0], [2, 0]])
    assert np.allclose(np_expect, ms_y.asnumpy()), f"np_expect:{np_expect}, ms_y:{ms_y}"


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_setitem_graph_mode(mode):
    """
    Feature: tensor setitem
    Description: Verify the result of tensor setitem in graph mode
    Expectation: success
    """
    ms.set_context(mode=mode)
    os.environ["MS_DEV_JIT_ENABLE_VIEW_OP"] = '1'
    np_x = np.arange(2 * 3).reshape((2, 3)).astype(np.float32)
    ms_x = Tensor(np_x)
    ms_x[0] = -1
    np_expect = np.array([[-1, -1, -1], [3, 4, 5]]).astype(np.float32)
    assert np.allclose(np_expect, ms_x.asnumpy()), f"np_expect:{np_expect}, ms_x:{ms_x}"


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
