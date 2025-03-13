# Copyright 2025 Huawei Technologies Co., Ltd
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
"""
Test Tensor.cross method.

This operator is tested by tests/st/ops/test_ops_cross.py
This file only tests the method call.
"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell


@run_with_cell
def cross(x, other, dim):
    return x.cross(other, dim)


@run_with_cell
def cross_named(x, other, dim):
    return x.cross(other=other, dim=dim)


@run_with_cell
def cross_grad(x, other, dim):
    return ms.grad(cross, (0, 1))(x, other, dim)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize(
    'mode, level',
    [
        (ms.PYNATIVE_MODE, 'O0'),
        (ms.GRAPH_MODE, 'O0'),
        (ms.GRAPH_MODE, 'O2'),
    ]
)
def test_tensor_cross(mode, level):
    """
    Feature: Tensor.cross
    Description: test Tensor.cross forward (named, positional) and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode, jit_level=level)

    x = Tensor(np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8], [3, 6, 9]]), dtype=ms.float32)
    other = Tensor(np.array([[9, 7, 5], [3, 2, 1], [4, 6, 8], [2, 7, 5]]), dtype=ms.float32)
    dim = 1
    expect_forward = np.array([[-11, 22, -11], [-6, 12, -6], [8, -16, 8], [-33, 3, 9]], dtype=np.float32)
    expect_grad_x = np.array([[2, -4, 2], [1, -2, 1], [-2, 4, -2], [2, 3, -5]], dtype=np.float32)
    expect_grad_other = np.array([[1, -2, 1], [1, -2, 1], [1, -2, 1], [3, -6, 3]], dtype=np.float32)

    out1 = cross(x, other, dim)
    np.testing.assert_allclose(out1.asnumpy(), expect_forward)

    out2 = cross_named(x, other, dim)
    np.testing.assert_allclose(out2.asnumpy(), expect_forward)

    grads = cross_grad(x, other, dim)
    np.testing.assert_allclose(grads[0].asnumpy(), expect_grad_x)
    np.testing.assert_allclose(grads[1].asnumpy(), expect_grad_other)
