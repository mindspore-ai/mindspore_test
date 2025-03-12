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

import numpy as np
import pytest
import mindspore as ms
from mindspore import mint
from mindspore import jit
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


@test_utils.run_with_cell
def forward_channel_shuffle_net(input_x, groups):
    net = mint.nn.ChannelShuffle(groups)
    return net(input_x)


@test_utils.run_with_cell
def forward_channel_shuffle_net_dyn(input_x):
    net = mint.nn.ChannelShuffle(2)
    return net(input_x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', ['pynative', 'KBK', 'GE'])
def test_channel_shuffle_normal(mode):
    """
    Feature: ChannelShuffle
    Description: Verify the result of mint.nn.ChannelShuffle
    Expectation: success
    """
    input_x = ms.Tensor(np.arange(16).reshape((1, 4, 2, 2)), dtype=ms.int32)
    groups = 2
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = forward_channel_shuffle_net(input_x, 2)
    elif mode == 'KBK':
        output = (jit(forward_channel_shuffle_net, jit_level="O0"))(input_x, groups)
    else:
        output = (jit(forward_channel_shuffle_net, backend="GE"))(input_x, groups)
    expect_out = np.array([[[[0, 1], [2, 3]], [[8, 9], [10, 11]],
                            [[4, 5], [6, 7]], [[12, 13], [14, 15]]]]).astype(np.int32)
    assert np.allclose(output.asnumpy(), expect_out)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_channel_shuffle_dynamic_shape():
    """
    Feature: Test ChannelShuffle with dynamic shape in graph mode.
    Description: call mint.nn.ChannelShuffle with valid input and index.
    Expectation: return the correct value.
    """
    x1 = ms.Tensor(np.arange(32).reshape((1, 2, 4, 2, 2)), dtype=ms.int32)
    x2 = ms.Tensor(np.arange(64).reshape((1, 8, 4, 2)), dtype=ms.int32)

    TEST_OP(forward_channel_shuffle_net_dyn, [[ms.Tensor(x1)], [ms.Tensor(x2)]], '', disable_yaml_check=True)
