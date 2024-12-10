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
import numpy as np
import mindspore.context as context
from mindspore import mint
from mindspore import Tensor
from mindspore.nn import Cell
from tests.mark_utils import arg_mark


class Net(Cell):
    def __init__(self, alpha, axis):
        super(Net, self).__init__()
        self.alpha = alpha
        self.axis = axis

    def construct(self, x0, x1):
        y0 = mint.add(x0, x1, alpha=self.alpha)
        y1 = mint.sum(y0, self.axis)
        return y1


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_fuse():
    """
    Feature: test dvm op fusion precision
    Description: pynative mode
    Expectation: the result match with the expected result
    """
    os.environ["MS_DEV_ENABLE_DVM"] = "1"
    context.set_context(mode=context.PYNATIVE_MODE)
    np.random.seed(1)
    x0 = np.random.normal(0, 1, (32, 128)).astype(np.float32)
    x1 = np.abs(np.random.normal(0, 1, (32, 128)).astype(np.float32))
    alpha = 2.3
    axis = (0,)
    expect = np.sum(x0 + x1 * alpha, axis)
    x0_ms = Tensor(x0)
    x1_ms = Tensor(x1)
    net = Net(alpha, axis)
    output = net(x0_ms, x1_ms)
    output = output.asnumpy()
    np.testing.assert_allclose(expect, output, 1e-4, 1e-4)
    os.environ.pop("MS_DEV_ENABLE_DVM")
