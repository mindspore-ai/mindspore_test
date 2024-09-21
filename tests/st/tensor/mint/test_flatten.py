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
import os
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
from tests.mark_utils import arg_mark
import random

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

class FlattenNet(nn.Cell):
    def construct(self, x, start_dim, end_dim, parm_mode=True):
        if os.environ.get("MS_TENSOR_API_ENABLE_MINT", "0") != '1':
            res = x.flatten(order='C', start_dim=start_dim, end_dim=end_dim)
        elif parm_mode:
            res = x.flatten(start_dim, end_dim)
        else:
            res = x.flatten(start_dim=start_dim, end_dim=end_dim)
        return res

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE])
def test_flatten_normal(mode):
    """
    Feature: Tensor.flatten.
    Description: Verify the result of flatten.
    Expectation: expect correct result.
    """
    os.environ["MS_TENSOR_API_ENABLE_MINT"] = '1'
    ms.set_context(mode=mode)
    test_shape = (2, 3, 4, 5)
    x = generate_random_input(test_shape, np.float32)
    tx = ms.Tensor(x, dtype=ms.float32)
    net = FlattenNet()
    output = net(tx, 1, 2)
    output2 = net(tx, 1, 2, False)
    expect = x.reshape((2, 12, 5))
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)
    np.testing.assert_allclose(output2.asnumpy(), expect, rtol=1e-4)

    output3 = net(ms.Tensor(x), 0, 2)
    output4 = net(ms.Tensor(x), 0, 2, False)
    expect3 = x.reshape((24, 5))
    np.testing.assert_allclose(output3.asnumpy(), expect3, rtol=1e-4)
    np.testing.assert_allclose(output4.asnumpy(), expect3, rtol=1e-4)
    del os.environ["MS_TENSOR_API_ENABLE_MINT"]

@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("mode", [ms.PYNATIVE_MODE])
def test_flatten_random(mode):
    """
    Feature: Tensor.flatten.
    Description: Verify the result of flatten.
    Expectation: expect correct result.
    """
    os.environ["MS_TENSOR_API_ENABLE_MINT"] = '1'
    def flatten_shape(shape, start, end):
        new_shape = []
        for i in range(start):
            new_shape.append(shape[i])
        flattened_dim_size = 1
        for i in range(start, end + 1):
            flattened_dim_size *= shape[i]
        new_shape.append(flattened_dim_size)
        for i in range(end + 1, len(shape)):
            new_shape.append(shape[i])
        return tuple(new_shape)

    ms.set_context(mode=mode)
    test_shape = (2, 3, 4, 5)
    x = generate_random_input(test_shape, np.float32)
    tx = ms.Tensor(x, dtype=ms.float32)
    net = FlattenNet()

    start_dim = [random.randint(0, 3) for _ in range(10)]
    end_dim = [random.randint(0, 3) for _ in range(10)]
    for i in range(10):
        if start_dim[i] > end_dim[i]:
            continue
        output = net(tx, start_dim[i], end_dim[i])
        output2 = net(tx, start_dim[i], end_dim[i], False)
        expect = x.reshape(flatten_shape(test_shape, start_dim[i], end_dim[i]))
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)
        np.testing.assert_allclose(output2.asnumpy(), expect, rtol=1e-4)
    del os.environ["MS_TENSOR_API_ENABLE_MINT"]
