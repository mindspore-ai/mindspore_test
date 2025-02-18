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
import numpy as np
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import mint
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_empty_normal1():
    """
    Feature: Ops.
    Description: test empty.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_size = (1, 2, 3)
    dtype = mstype.float32

    y = mint.empty(input_size)
    np.testing.assert_equal(y.shape, input_size)
    np.testing.assert_equal(y.dtype, dtype)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_normal2():
    """
    Feature: Ops.
    Description: test empty.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_size = (1, 2, 3)
    dtype = mstype.float32

    y = mint.empty(input_size, device="CPU")
    np.testing.assert_equal(y.shape, input_size)
    np.testing.assert_equal(y.dtype, dtype)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_normal3():
    """
    Feature: Ops.
    Description: test empty.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_size = (1, 2, 3)
    dtype = mstype.float64

    y = mint.empty(input_size, dtype=dtype, device="Ascend")
    np.testing.assert_equal(y.shape, input_size)
    np.testing.assert_equal(y.dtype, dtype)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_normal4():
    """
    Feature: Ops.
    Description: test empty.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    input_size = (1, 2, 3, 4, 5, 6, 7)
    dtype = mstype.int64

    y = mint.empty(input_size, dtype=dtype)
    np.testing.assert_equal(y.shape, input_size)
    np.testing.assert_equal(y.dtype, dtype)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_normal5():
    """
    Feature: Ops.
    Description: test empty.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    dtype = mstype.float32

    y = mint.empty(1, 2, 3)
    np.testing.assert_equal(y.shape, (1, 2, 3))
    np.testing.assert_equal(y.dtype, dtype)

def empty_forward_func_dyn_test(input_size, dtype=None):
    y = mint.empty(input_size, dtype=dtype)
    return y.shape

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential')
def test_empty_dynamic_shape():
    """
    Feature: Test empty with dynamic shape in graph mode.
    Description: call ops.extend.empty with valid input and index.
    Expectation: return the correct value.
    """
    TEST_OP(empty_forward_func_dyn_test, [[(2, 3)], [(3, 4, 5)]], '', disable_yaml_check=True,
            disable_grad=True, disable_mode=['GRAPH_MODE', 'GRAPH_MODE_O0'])
