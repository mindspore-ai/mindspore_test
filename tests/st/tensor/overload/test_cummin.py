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
Test Tensor.cummin method.

This interface's operators are tested by:

- tests/st/ops/test_ops_cummin.py
- tests/st/dynamic_shape/test_cummin.py

This file only tests the method call.
"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from tests.mark_utils import arg_mark
from tests.st.utils.test_utils import run_with_cell


@run_with_cell
def cummin(x, dim):
    return x.cummin(dim)


@run_with_cell
def cummin_named_dim(x, dim):
    return x.cummin(dim=dim)


@run_with_cell
def cummin_grad(f, x, dim):
    return ms.grad(f, (0, 1))(x, dim)


def _test_tensor_cummin_main(forward_net, grad_check: bool):
    x = Tensor(np.array([[3, 1, 4, 1], [1, 5, 9, 2]], dtype=np.float32))
    dim = -2
    expect_values = np.array([[3, 1, 4, 1], [1, 1, 4, 1]], dtype=np.float32)
    expect_indices = np.array([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int64)
    expect_grad = np.array([[1, 2, 2, 2], [1, 0, 0, 0]], dtype=np.float32)

    values, indices = forward_net(x, dim)
    assert values.dtype == ms.float32
    assert np.allclose(values.asnumpy(), expect_values)
    assert indices.dtype == ms.int64
    assert np.allclose(indices.asnumpy(), expect_indices)

    if (ms.get_context('device_target') == 'Ascend') and grad_check:
        output_grad = cummin_grad(forward_net, x, dim)
        assert np.allclose(output_grad.asnumpy(), expect_grad)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize(
    'mode, level',
    [
        (ms.PYNATIVE_MODE, 'O0'),
        (ms.GRAPH_MODE, 'O0'),
        (ms.GRAPH_MODE, 'O2'),
    ]
)
def test_tensor_cummin(mode, level):
    """
    Feature: Tensor.cummin(dim)
    Description: test Tensor.cummin(dim) forward (named, positional) and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode, jit_level=level)

    grad_check = level != 'O2'
    _test_tensor_cummin_main(cummin, grad_check)
    _test_tensor_cummin_main(cummin_named_dim, grad_check)
