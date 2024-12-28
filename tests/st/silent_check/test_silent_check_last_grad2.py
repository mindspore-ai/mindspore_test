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
"""
Test module for testing silent check.
"""
import os
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops, jit
from mindspore.ops import operations as op
from mindspore.common import mutable
from tests.mark_utils import arg_mark
import pytest

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", ["kbk", "pyboost"])
def test_silent_check_grad_of_all(mode):
    """
    Feature: Test silent check for gradient of all inputs and parameters in pynative mode
    Description: Test silent check for gradient of all inputs and parameters in pynative mode
    Expectation: No errors occurs when enable silent check
    """
    os.environ['NPU_ASD_ENABLE'] = '1'
    if mode == 'pyboost':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE)
    input_tensor = Tensor(np.random.randn(32, 2048).astype(np.float32))
    net = nn.Dense(in_channels=2048, out_channels=12, weight_init='ones', bias_init='zeros', activation='relu')
    net.set_train()
    output = net(input_tensor)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True, sens_param=True)
    grad_net = grad_op(net)
    grads = grad_net(input_tensor, output)
    assert len(grads) == 2
    assert len(grads[0]) == 1
    assert grads[0][0].asnumpy().shape == (32, 2048)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("mode", ["pyboost"])
def test_avg_pool_grad(mode):
    """
    Feature: Test silent check for last grad op is avg_pool
    Description: TTest silent check for last grad op is avg_pool
    Expectation: No errors occurs when enable silent check
    """
    os.environ['NPU_ASD_ENABLE'] = '2'
    if mode == 'pyboost':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.avg_pool = op.AvgPool(kernel_size=2, strides=1, pad_mode='same', data_format='NCHW')

        def construct(self, input_x):
            return self.avg_pool(input_x)

    input_tensor = Tensor(np.random.randn(2, 3, 4, 5).astype(np.float32))
    net = Net()
    output = net(input_tensor)
    grad_op = ops.GradOperation(get_all=True, sens_param=True)
    grad_net = grad_op(net)
    grads = grad_net(input_tensor, output)
    assert len(grads) == 1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_dynamic_shape_mutable():
    """
    Feature: Test silent check for mutalbe input
    Description: TTest silent check for for mutalbe input
    Expectation: No errors occurs when enable silent check
    """
    os.environ['NPU_ASD_ENABLE'] = '2'
    ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    @jit
    def add_xy(x, y):
        out = x + y
        return out

    x = (2, 3)
    y = (2, 3, 4)
    x = mutable(x, dynamic_len=True)
    out = add_xy(x, y)
    assert out == (2, 3, 2, 3, 4)
