# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.functional import vmap

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class OpNetWrapper(nn.Cell):
    def __init__(self, op):
        super(OpNetWrapper, self).__init__()
        self.op = op

    def construct(self, *inputs):
        return self.op(*inputs)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_case1_basic_func():
    op = P.GatherNd()
    op_wrapper = OpNetWrapper(op)

    indices = Tensor(np.array([[0, 0], [1, 1]]), mindspore.int32)
    params = Tensor(np.array([[0, 1], [2, 3]]), mindspore.float32)
    outputs = op_wrapper(params, indices)
    print(outputs)
    expected = [0, 3]
    assert np.allclose(outputs.asnumpy(), np.array(expected))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_case2_indices_to_matrix():
    op = P.GatherNd()
    op_wrapper = OpNetWrapper(op)

    indices = Tensor(np.array([[1], [0]]), mindspore.int32)
    params = Tensor(np.array([[0, 1], [2, 3]]), mindspore.float32)
    outputs = op_wrapper(params, indices)
    print(outputs)
    expected = [[2, 3], [0, 1]]
    assert np.allclose(outputs.asnumpy(), np.array(expected))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_case3_indices_to_3d_tensor():
    op = P.GatherNd()
    op_wrapper = OpNetWrapper(op)

    indices = Tensor(np.array([[1]]), mindspore.int32)  # (1, 1)
    params = Tensor(np.array([[[0, 1], [2, 3]],
                              [[4, 5], [6, 7]]]), mindspore.float32)  # (2, 2, 2)
    outputs = op_wrapper(params, indices)
    print(outputs)
    expected = [[[4, 5], [6, 7]]]  # (1, 2, 2)
    assert np.allclose(outputs.asnumpy(), np.array(expected))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_case4():
    op = P.GatherNd()
    op_wrapper = OpNetWrapper(op)

    indices = Tensor(np.array([[0, 1], [1, 0]]), mindspore.int32)  # (2, 2)
    params = Tensor(np.array([[[0, 1], [2, 3]],
                              [[4, 5], [6, 7]]]), mindspore.float32)  # (2, 2, 2)
    outputs = op_wrapper(params, indices)
    print(outputs)
    expected = [[2, 3], [4, 5]]  # (2, 2)
    assert np.allclose(outputs.asnumpy(), np.array(expected))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_case5():
    op = P.GatherNd()
    op_wrapper = OpNetWrapper(op)

    indices = Tensor(np.array([[0, 0, 1], [1, 0, 1]]), mindspore.int32)  # (2, 3)
    params = Tensor(np.array([[[0, 1], [2, 3]],
                              [[4, 5], [6, 7]]]), mindspore.float32)  # (2, 2, 2)
    outputs = op_wrapper(params, indices)
    print(outputs)
    expected = [1, 5]  # (2,)
    assert np.allclose(outputs.asnumpy(), np.array(expected))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_case6():
    op = P.GatherNd()
    op_wrapper = OpNetWrapper(op)

    indices = Tensor(np.array([[[0, 0]], [[0, 1]]]), mindspore.int32)  # (2, 1, 2)
    params = Tensor(np.array([[[0, 1], [2, 3]],
                              [[4, 5], [6, 7]]]), mindspore.float32)  # (2, 2, 2)
    outputs = op_wrapper(params, indices)
    print(outputs)
    expected = [[[0, 1]], [[2, 3]]]  # (2, 1, 2)
    assert np.allclose(outputs.asnumpy(), np.array(expected))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_case7():
    op = P.GatherNd()
    op_wrapper = OpNetWrapper(op)

    indices = Tensor(np.array([[[1]], [[0]]]), mindspore.int32)  # (2, 1, 1)
    params = Tensor(np.array([[[0, 1], [2, 3]],
                              [[4, 5], [6, 7]]]), mindspore.float32)  # (2, 2, 2)
    outputs = op_wrapper(params, indices)
    print(outputs)
    expected = [[[[4, 5], [6, 7]]], [[[0, 1], [2, 3]]]]  # (2, 1, 2, 2)
    assert np.allclose(outputs.asnumpy(), np.array(expected))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_case8():
    op = P.GatherNd()
    op_wrapper = OpNetWrapper(op)

    indices = Tensor(np.array([[[0, 1], [1, 0]], [[0, 0], [1, 1]]]), mindspore.int32)  # (2, 2, 2)
    params = Tensor(np.array([[[0, 1], [2, 3]],
                              [[4, 5], [6, 7]]]), mindspore.float32)  # (2, 2, 2)
    outputs = op_wrapper(params, indices)
    print(outputs)
    expected = [[[2, 3], [4, 5]], [[0, 1], [6, 7]]]  # (2, 2, 2)
    assert np.allclose(outputs.asnumpy(), np.array(expected))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
def test_case9():
    op = P.GatherNd()
    op_wrapper = OpNetWrapper(op)

    indices = Tensor(np.array([[[0, 0, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 0]]]), mindspore.int32)  # (2, 2, 3)
    params = Tensor(np.array([[[0, 1], [2, 3]],
                              [[4, 5], [6, 7]]]), mindspore.int64)  # (2, 2, 2)
    outputs = op_wrapper(params, indices)
    print(outputs)
    expected = [[1, 5], [3, 6]]  # (2, 2, 2)
    assert np.allclose(outputs.asnumpy(), np.array(expected))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_gather_nd_vmap():
    """
    Feature: gather_nd vmap test on cpu.
    Description: test the rightness of gathernd vmap feature.
    Expectation: use vmap rule's result equal to manually batched.
    """

    def cal_gather_nd(x, index):
        return P.GatherNd()(x, index)

    x = Tensor(np.array([[[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                          [[2, 2, 2, 2], [2, 2, 2, 2], [2, 2, 2, 2]]],
                         [[[3, 3, 3, 3], [3, 3, 3, 3], [3, 3, 3, 3]],
                          [[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]]]]).astype(np.float32))

    y = Tensor(np.array([[[[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                          [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
                         [[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                          [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]]]).astype(np.int32))

    vmap_gather_nd = vmap(vmap(cal_gather_nd, in_axes=(-1, -1)), in_axes=(-1, -1))
    outputs = vmap_gather_nd(x, y)
    expect = np.array([[[1, 3], [1, 3], [1, 3]], [[1, 3], [1, 3], [1, 3]], [
        [1, 3], [1, 3], [1, 3]], [[1, 3], [1, 3], [1, 3]]]).astype(np.float32)
    assert np.allclose(outputs.asnumpy(), expect)


if __name__ == '__main__':
    test_case1_basic_func()
    test_case2_indices_to_matrix()
    test_case3_indices_to_3d_tensor()
    test_case4()
    test_case5()
    test_case6()
    test_case7()
    test_case8()
    test_case9()
