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
import mindspore.mint.nn as nn
from mindspore import Tensor
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.ops.ops_binary_cases import ops_binary_cases, OpsBinaryCase


@test_utils.run_with_cell
def forward_adaptive_avg_pool1d_net(input_x, output_size):
    net = nn.AdaptiveAvgPool1d(output_size)
    return net(input_x)


@test_utils.run_with_cell
def forward_adaptive_avg_pool1d_for_dyn(input_x):
    net = nn.AdaptiveAvgPool1d(8)
    return net(input_x)


@test_utils.run_with_cell
def grad_adaptive_avg_pool1d_net(input_x, output_size):
    net = nn.AdaptiveAvgPool1d(output_size)
    return ms.grad(net)(input_x)


@ops_binary_cases(OpsBinaryCase(input_info=[((16, 4, 16), np.float16)],
                                output_info=[((16, 4, 8), np.float16), ((16, 4, 16), np.float16)],
                                extra_info='AdaptiveAvgPool1d'))
def nn_adaptive_avg_pool1d_case1(input_binary_data=None, output_binary_data=None):
    output = forward_adaptive_avg_pool1d_net(Tensor(input_binary_data[0]), 8)
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-03, 1e-03)
    output = grad_adaptive_avg_pool1d_net(Tensor(input_binary_data[0]), 8)
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-03, 1e-03)


@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_adaptive_avg_pool1d(mode):
    """
    Feature: adaptive_avg_pool1d
    Description: Verify the result of adaptive_avg_pool1d.
    Expectation: success
    """

    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(jit_config={"jit_level": "O0"}, mode=ms.GRAPH_MODE)
    nn_adaptive_avg_pool1d_case1()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
def test_adaptive_avg_pool1d_dyn():
    """
    Feature: Dynamic shape of adaptive_avg_pool1d
    Description: test adaptive_avg_pool1d with dynamic rank/shape.
    Expectation: success
    """
    in1 = Tensor(np.random.randn(4, 4, 3), dtype=ms.float32)
    in2 = Tensor(np.random.randn(2, 4), dtype=ms.float32)
    TEST_OP(forward_adaptive_avg_pool1d_for_dyn, [[in1], [in2]], '', disable_yaml_check=True,
            disable_mode=['GRAPH_MODE'])
