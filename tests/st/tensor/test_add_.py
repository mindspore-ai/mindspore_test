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
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.mark_utils import arg_mark

class Net(ms.nn.Cell):
    def construct(self, x, y, alpha=1):
        x.add_(y, alpha=alpha)
        return x


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def inplace_add_forward_func(x, y, alpha=1):
    return Net()(x, y, alpha)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_inplace_add_std():
    """
    Feature: standard forward, backward features.
    Description: test function copy.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    y = generate_random_input((2, 3, 4), np.float32)
    z = generate_random_input((2, 3, 1), np.float32)  # broadcast

    alpha = 1.5

    expect_y = x + y * alpha
    expect_z = x + z * alpha

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output_y = inplace_add_forward_func(ms.Tensor(x), ms.Tensor(y), alpha)
    output_z = inplace_add_forward_func(ms.Tensor(x), ms.Tensor(z), alpha)

    np.allclose(output_y.asnumpy(), expect_y, rtol=1e-5, equal_nan=True)
    np.allclose(output_z.asnumpy(), expect_z, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_copy_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test copy forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_y1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    tensor_y2 = ms.Tensor(generate_random_input((1, 1, 5), np.float32))  # broadcast
    alpha1 = 1.5
    alpha2 = 2.5

    TEST_OP(inplace_add_forward_func, [[tensor_x1, tensor_y1, alpha1], [tensor_x2, tensor_y2, alpha2]],
            'inplace_add_ext', disable_mode=['GRAPH_MODE', 'GRAPH_MODE_O0'])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_copy_bfloat16():
    """
    Feature: test copy functional API.
    Description: testcase for copy functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((3, 4, 2), np.float32)
    y = generate_random_input((3, 4, 1), np.float32)

    expect_y = x + y

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output = inplace_add_forward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))

    np.allclose(output.float().asnumpy(), expect_y, 0.004, 0.004, equal_nan=True)
