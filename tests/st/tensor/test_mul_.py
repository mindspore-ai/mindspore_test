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
from tests.mark_utils import arg_mark

class Net(ms.nn.Cell):
    def construct(self, x, y):
        x.mul_(y)
        return x


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def inplace_mul_forward_func(x, y):
    return Net()(x, y)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_inplace_mul_std():
    """
    Feature: standard forward, backward features.
    Description: test function copy.
    Expectation: expect correct result.
    """
    x = generate_random_input((2, 3, 4), np.float32)
    y = generate_random_input((2, 3, 4), np.float32)
    z = generate_random_input((2, 3, 1), np.float32)  # broadcast

    expect_mul_xy = x * y
    expect_mul_xz = x * z

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output_y = inplace_mul_forward_func(ms.Tensor(x), ms.Tensor(y))
    output_z = inplace_mul_forward_func(ms.Tensor(x), ms.Tensor(z))

    np.allclose(output_y.asnumpy(), expect_mul_xy, rtol=1e-5, equal_nan=True)
    np.allclose(output_z.asnumpy(), expect_mul_xz, rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_copy_bfloat16():
    """
    Feature: test copy functional API.
    Description: testcase for copy functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((3, 4, 2), np.float32)
    y = generate_random_input((3, 4, 1), np.float32)

    expect_y = x * y

    ms.context.set_context(mode=ms.PYNATIVE_MODE)
    output = inplace_mul_forward_func(ms.Tensor(x, dtype=ms.bfloat16), ms.Tensor(y, dtype=ms.bfloat16))

    np.allclose(output.float().asnumpy(), expect_y, 0.004, 0.004, equal_nan=True)
