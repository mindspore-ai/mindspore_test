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

import pytest
import numpy as np
from numpy.testing import assert_allclose

import mindspore as ms
from mindspore.ops import GradOperation
from mindspore import Tensor, nn

from tests.st.utils import test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark


def _assert_equals(result: Tensor, slf: Tensor, exp: np.ndarray, delta):
    assert result.dtype == slf.dtype
    assert_allclose(result.asnumpy().astype(np.float32), exp, delta, delta, equal_nan=True)
    # inplace operation: self and returns should be the same tensor
    assert np.array_equal(slf.asnumpy().astype(np.float32), result.asnumpy().astype(np.float32), equal_nan=True)
    result[0, 0, 0] += np.random.rand() * 8
    assert result[0, 0, 0] == slf[0, 0, 0]


@test_utils.run_with_cell
def sqrt(x):
    return x.sqrt_()


@test_utils.run_with_cell
def sqrt_with_grad(x):
    return (x * Tensor(1, dtype=x.dtype)).sqrt_()


class SqrtGrad(nn.Cell):
    def __init__(self, sens: Tensor):
        super().__init__()
        self.grad_op = GradOperation(sens_param=True)
        self.grad_wrt_output = sens

    def construct(self, x):
        return self.grad_op(sqrt_with_grad)(x, self.grad_wrt_output)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_inplace_sqrt(mode):
    """
    Feature: Tensor ops.
    Description: test op sqrt_ with PYNATIVE mode support (sqrt_ not supports GRAPH MODE).
    Expectation: expect correct result.
    """
    _test_inplace_scatter_src_main(mode, ms.float32)


def _test_inplace_scatter_src_main(mode, input_type):
    ms.context.set_context(mode=mode, jit_level='O0', device_target="Ascend")
    delta = 4e-3 if input_type == ms.bfloat16 else 1e-5
    ## forward
    x_np = np.arange(2*3*4).reshape(2, 3, 4) / 4.0
    out_exp = x_np.copy()
    x_np = x_np ** 2

    x = Tensor(x_np.copy(), dtype=input_type)
    result = x.sqrt_()
    _assert_equals(result, x, out_exp, delta)

    ## inplace backward
    x = Tensor(x_np.copy(), dtype=input_type)
    dy = np.arange(2*3*4).reshape(2, 3, 4) / 2 + 1
    dx_exp = dy.copy() / 2
    dx_exp /= out_exp
    dx = SqrtGrad(Tensor(dy, dtype=input_type))(x)
    assert_allclose(dx.asnumpy(), dx_exp, delta, delta, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_sqrt_bfloat16(mode):
    """
    Feature: Tensor ops.
    Description: test op sqrt_ with PYNATIVE mode support (sqrt_ not supports GRAPH MODE).
    Expectation: expect correct result.
    """
    _test_inplace_scatter_src_main(mode, ms.bfloat16)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_inplace_scatter_dynamic():
    """
    Feature: test dynamic by TEST_OP.
    Description: test ops.sqrt_ dynamic shape feature.
    Expectation: expect correct result.
    """
    x1 = Tensor(np.array([[[[0.6777, 3.8882, 1.4999, 2.4321]]]], dtype=np.float32))
    x2 = Tensor(np.array([[1.2333, 2.6667, 3], [4.8, 5.12, -6.5536], [3.59, 7.87, -0.919]], dtype=np.float32))

    args = ([[x1], [x2]], 'inplace_sqrt')
    kwargs = dict(
        disable_mode=['GRAPH_MODE'],   # not support yet
        inplace_update=True,
    )
    TEST_OP(sqrt, *args, **kwargs, disable_grad=True)  # grad is tested by next TEST_OP
    TEST_OP(sqrt_with_grad, *args, **kwargs)
