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
from mindspore import mint
from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


@test_utils.run_with_cell
def histc_forward_func(x, bins, min_val, max_val):
    return mint.histc(x, bins, min_val, max_val)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_ops_histc_ext_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function histc forward.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor([1, 2, 1], ms.int32)
    bins, min_val, max_val = 4, 0.0, 3.0
    output = histc_forward_func(x, bins, min_val, max_val)
    expected_output = np.array([0, 2, 1, 0])
    assert np.array_equal(output.asnumpy(), expected_output)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'],
          level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_histc_ext_dynamic_shape():
    """
    Feature: Test operator Histc by TEST_OP
    Description:  Test operator Histc with dynamic input
    Expectation: the result of Histc is correct.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.int32)
    x2 = generate_random_input((3, 4, 5, 6, 7), np.int32)
    TEST_OP(histc_forward_func, [[ms.Tensor(x1), 4, 0.0, 3.0], [ms.Tensor(x2), 5, 1.0, 8.0]],
            'histc_ext', disable_yaml_check=True, disable_grad=True, disable_mode=["GRAPH_MODE"])
