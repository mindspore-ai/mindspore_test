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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import mint, Tensor
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_empty_like_normal1():
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    y = mint.empty_like(input_tensor)
    np.testing.assert_equal(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float32)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_like_normal2():
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    y = mint.empty_like(input_tensor, dtype=mstype.float64)
    np.testing.assert_equal(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float64)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_like_normal3():
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    y = mint.empty_like(input_tensor, dtype=mstype.float64, device="Ascend")
    np.testing.assert_equal(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float64)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_like_normal4():
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    y = mint.empty_like(input_tensor, dtype=mstype.float64, device="CPU")
    np.testing.assert_equal(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float64)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_like_normal5():
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = mint.ones((1, 2, 3))

    y = mint.empty_like(input_tensor)
    np.testing.assert_equal(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, input_tensor.dtype)

def empty_like_forward_func_dyn_test(input_tensor, dtype=None):
    y = mint.empty_like(input_tensor, dtype=dtype)
    return y.shape

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_like_dynamic_shape():
    """
    Feature: Test empty_like with dynamic shape in graph mode.
    Description: call ops.extend.empty_like with valid input and index.
    Expectation: return the correct value.
    """
    tensor_1 = Tensor(np.arange(6).reshape(2, 3), dtype=mstype.float32)

    tensor_2 = Tensor(np.arange(24).reshape(2, 3, 4), dtype=mstype.float32)

    TEST_OP(empty_like_forward_func_dyn_test, [[tensor_1], [tensor_2]], '', disable_yaml_check=True,
            disable_grad=True, disable_mode=['GRAPH_MODE', 'GRAPH_MODE_O0'])

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("device_name", ['npu', 'ascend', 'cpu', 'Ascend', 'CPU'])
def test_empty_like_device(device_name):
    """
    Feature: Ops.
    Description: test empty_like.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_tensor = Tensor(np.arange(6).reshape(1, 2, 3), dtype=mstype.float32)

    y = mint.empty_like(input_tensor, device=device_name)
    np.testing.assert_equal(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, mstype.float32)
