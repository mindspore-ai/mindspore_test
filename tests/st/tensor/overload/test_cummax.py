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
Test Tensor.cummax method.

This operator is tested by:

- tests/st/dynamic_shape/test_cummax.py
- tests/st/ops/test_ops_cummax.py

This file only tests the method call.
"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell


@run_with_cell
def cummax(x, dim):
    return x.cummax(dim)


@run_with_cell
def cummax_named(x, dim):
    return x.cummax(dim=dim)


@run_with_cell
def cummax_grad(x, dim):
    return ms.grad(cummax, (0, 1))(x, dim)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_tensor_cummax(mode):
    """
    Feature: Tensor.cummax
    Description: test Tensor.cummax forward (named, positional) and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode, jit_level='O0')

    x = Tensor(np.array([[1, 4, 3, 2], [7, 6, 5, 8]], dtype=np.float32))
    dim = -2
    expect_values = np.array([[1, 4, 3, 2], [7, 6, 5, 8]], dtype=np.float32)
    expect_indices = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int64)
    expect_grad = np.array([[1, 1, 1, 1], [1, 1, 1, 1]], dtype=np.float32)

    output = cummax(x, dim)
    assert np.allclose(output[0].asnumpy(), expect_values)
    assert np.allclose(output[1].asnumpy(), expect_indices)

    output = cummax_named(x, dim)
    assert np.allclose(output[0].asnumpy(), expect_values)
    assert np.allclose(output[1].asnumpy(), expect_indices)

    if ms.get_context('device_target') == 'Ascend':
        output_grad = cummax_grad(x, dim)
        assert np.allclose(output_grad.asnumpy(), expect_grad)
