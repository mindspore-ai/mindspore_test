# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import ops
from mindspore.ops.auto_generate import contiguous
from mindspore import Tensor
import numpy as np
from tests.mark_utils import arg_mark


def test_contiguous_contiguous():
    """
    Feature: test cast operator
    Description: test contiguous run by pyboost
    Expectation: success
    """
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
    output = contiguous(x)

    assert np.allclose(output.asnumpy(), x.asnumpy())


def test_contiguous_dim2_with_transpose():
    """
    Feature: test cast operator
    Description: test contiguous run by pyboost
    Expectation: success
    """
    x = Tensor(np.random.randn(2, 3).astype(np.float32))
    y = ops.transpose(x, (1, 0))
    output = contiguous(y)

    transpose_result = ops.transpose(x, (1, 0))
    assert np.allclose(output.asnumpy(), transpose_result.asnumpy())


def test_contiguous_dim3_with_transpose():
    """
    Feature: test cast operator
    Description: test contiguous run by pyboost
    Expectation: success
    """
    x = Tensor(np.random.randn(2, 3, 4).astype(np.float32))
    y = ops.transpose(x, (2, 0, 1))
    output = contiguous(y)

    transpose_result = ops.transpose(x, (2, 0, 1))
    assert np.allclose(output.asnumpy(), transpose_result.asnumpy())


def test_contiguous_dim4_with_transpose():
    """
    Feature: test cast operator
    Description: test contiguous run by pyboost
    Expectation: success
    """
    x = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    y = ops.transpose(x, (2, 0, 3, 1))
    output = contiguous(y)

    transpose_result = ops.transpose(x, (2, 0, 3, 1))
    assert np.allclose(output.asnumpy(), transpose_result.asnumpy())


def test_contiguous_dim2_with_slice():
    """
    Feature: test cast operator
    Description: test contiguous run by pyboost
    Expectation: success
    """
    x = Tensor(np.random.randn(4, 5).astype(np.float32))
    y = x[1:3:2, 1:4:2]
    output = contiguous(y)

    slice_res = x[1:3:2, 1:4:2]
    assert np.allclose(output.asnumpy(), slice_res.asnumpy())


def test_contiguous_dim3_with_slice():
    """
    Feature: test cast operator
    Description: test contiguous run by pyboost
    Expectation: success
    """
    x = Tensor(np.random.randn(4, 5, 6).astype(np.float32))
    y = x[1:3:2, 1:4:2, 2:6:2]
    output = contiguous(y)

    slice_res = x[1:3:2, 1:4:2, 2:6:2]
    assert np.allclose(output.asnumpy(), slice_res.asnumpy())


def test_contiguous_dim4_with_slice():
    """
    Feature: test cast operator
    Description: test contiguous run by pyboost
    Expectation: success
    """
    x = Tensor(np.random.randn(4, 5, 6, 4).astype(np.float32))
    y = x[1:3:2, 1:4:2, 2:6:2, 1:4:2]
    output = contiguous(y)

    slice_res = x[1:3:2, 1:4:2, 2:6:2, 1:4:2]
    assert np.allclose(output.asnumpy(), slice_res.asnumpy())

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential')
def test_ops_contiguous():
    """
    Feature: test contiguous
    Description: test contiguous
    Expectation: success
    """
    test_contiguous_contiguous()
    test_contiguous_dim2_with_transpose()
    test_contiguous_dim3_with_transpose()
    test_contiguous_dim4_with_transpose()
    test_contiguous_dim2_with_slice()
    test_contiguous_dim3_with_slice()
    test_contiguous_dim4_with_slice()
