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
"""test nn.module in PIJIT"""

import torch
import mindspore as ms
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_nn_module():
    """
    Feature: PIJIT
    Description: Test torch.nn.Module
    Expectation: success
    """
    class Net(torch.nn.Module):
      def forward(self, x):
        return x + 1

    net = Net()

    @ms.jit(capture_mode="bytecode", fullgraph=True)
    def func(x):
        return net(x)

    ms.set_context(mode=ms.PYNATIVE_MODE)
    assert func(1) == 2
