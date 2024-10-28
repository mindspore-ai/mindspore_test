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
import mindspore as ms
import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.common.api import _pynative_executor
from mindspore.ops.function.array_func import expand_as
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def expand_as_forward_func(x, other_tensor):
    return expand_as(x, other_tensor)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_expand_as(mode):
    """
    Feature: pyboost function.
    Description: test function expand_as forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    shape = (4, 5, 2, 3, 4, 5, 6)
    input_np = np.random.rand(2, 3, 1, 5, 1).astype(np.float32)
    other_np = np.random.rand(*shape).astype(np.float32)
    output = expand_as(Tensor(input_np), Tensor(other_np))
    expect = np.broadcast_to(input_np, shape)
    assert np.allclose(output.asnumpy(), expect)

    shape = (3, 5, 7, 4, 5, 6)
    input_np = np.arange(20).reshape((4, 5, 1)).astype(np.int32)
    other_np = np.random.rand(*shape).astype(np.int32)
    output = expand_as(Tensor(input_np), Tensor(other_np))
    expect = np.broadcast_to(input_np, shape)
    assert np.allclose(output.asnumpy(), expect)


def expand_as_dtype(dtype):
    """
    Basic function to test data type of ExpandAs.
    """
    shape = (2, 3, 4, 5)
    other_np = np.random.rand(*shape).astype(dtype)
    input_np = np.random.rand(4, 5).astype(dtype)
    output = expand_as(Tensor(input_np), Tensor(other_np))
    expect = np.broadcast_to(input_np, shape)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_expand_as_dtype(mode):
    """
    Feature: Test supported data types of ExpandAs.
    Description: all data types
    Expectation: success.
    """
    context.set_context(mode=mode)
    types = [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
             np.uint8, np.uint16, np.uint32, np.uint64, np.complex64, np.complex128]
    for dtype in types:
        expand_as_dtype(dtype=dtype)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_expand_as_exception(mode):
    """
    Feature: Test invalid input and target shape in of ExpandAs.
    Description: target shape is empty, but input shape is not empty.
    Expectation: the result match with expected result.
    """
    with pytest.raises(Exception) as info:
        context.set_context(mode=mode)
        other_np = np.random.rand().astype(np.float32)
        input_np = np.random.rand(3, 4).astype(np.float32)
        expand_as(Tensor(input_np), Tensor(other_np))
        _pynative_executor.sync()
        assert "ValueError: For 'ExpandAs', each dimension pair, input_x shape and target shape must be equal or \
        input dimension is 1 or target dimension is -1. But got input_x shape: [const vector][], target shape: \
        [const vector][0]." in str(info.value)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_expand_as_forward(mode):
    """
    Feature: Ops.
    Description: test expand_as.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    shape = (128, 1, 77, 77)
    input_np = np.arange(128).reshape((128, 1, 1, 1)).astype(np.float32)
    other_np = np.random.rand(*shape).astype(np.float32)
    out = expand_as_forward_func(Tensor(input_np), Tensor(other_np))
    expect = np.broadcast_to(input_np, shape)
    assert np.allclose(out.asnumpy(), expect)
