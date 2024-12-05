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
import pytest
import numpy as np
from tests.st.utils import test_utils

import mindspore as ms
from mindspore import Tensor, context
from mindspore import ops


def conv2d(x, weight):
    op = ops.Conv2D(out_channel=32, kernel_size=3)
    return op(x, weight)


@test_utils.run_with_cell
def conv2d_forward_func(x, weight):
    return conv2d(x, weight)


@test_utils.run_with_cell
def conv2d_backward_func(x, weight):
    return ms.grad(conv2d_forward_func, (0, 1))(x, weight)


@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_conv2d_forward(mode):
    """
    Feature: Ops
    Description: test op conv2d and auto grad of op conv2d
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    x = Tensor(np.ones([10, 32, 32, 32]), ms.bfloat16)
    weight = Tensor(np.ones([32, 32, 3, 3]), ms.bfloat16)
    output = conv2d_forward_func(x, weight)
    assert output.shape == (10, 32, 30, 30)

    ## auto grad
    x = Tensor(np.ones([10, 32, 32, 32]), ms.bfloat16)
    weight = Tensor(np.ones([32, 32, 3, 3]), ms.bfloat16)
    grads = conv2d_backward_func(x, weight)
    dx, dw = grads
    print(f"dx: {dx}\ndw: {dw}")
