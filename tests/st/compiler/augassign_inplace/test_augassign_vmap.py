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

"""inplace augassign vmap test cases."""

import pytest
from mindspore import nn
from mindspore import Tensor, jit, context
from mindspore.common import dtype
from mindspore.ops.functional import vmap
from mindspore._extends.parse import compile_config
import numpy as np
from tests.mark_utils import arg_mark


@pytest.fixture(scope="module", autouse=True)
def setup_teardown():
    compile_config.JIT_ENABLE_AUGASSIGN_INPLACE = '1'
    yield
    compile_config.JIT_ENABLE_AUGASSIGN_INPLACE = '0'


class AddNet(nn.Cell):
    def construct(self, x, y):
        out = y
        out += y
        x += 2
        return out, x


class SubNet(nn.Cell):
    def construct(self, x, y):
        out = y
        out -= y
        x -= 2
        return out, x


class MulNet(nn.Cell):
    def construct(self, x, y):
        out = y
        out *= y
        x *= 2
        return out, x


class DivNet(nn.Cell):
    def construct(self, x, y):
        out = y
        out /= y
        x /= 2
        return out, x


class FloorDivideNet(nn.Cell):
    def construct(self, x, y):
        out = y
        out //= y
        x //= 2
        return out, x


class ModNet(nn.Cell):
    def construct(self, x, y):
        out = y
        out %= y
        x %= 2
        return out, x


class ControlFlowNet(nn.Cell):
    def construct(self, x, y):
        out = y
        while x < 2:
            out = y + out
            x += 1
        return out


class VmapControlFlowNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.net = ControlFlowNet()
        self.vmap = vmap(self.net, in_axes=(None, 1), out_axes=0)

    def construct(self, x, y):
        return self.vmap(x, y)


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("network", [AddNet, SubNet, MulNet, DivNet, FloorDivideNet, ModNet])
def test_augassign_add_vmap(network):
    """
    Feature: Support augassign inplace vmap.
    Description: Support augassign inplace vmap.
    Expectation: Run success.
    """
    net = network()
    net.construct = jit(net.construct, backend='ms_backend')

    x = Tensor([1], dtype.float32)
    y = Tensor(np.ones([3, 4]), dtype.float32)
    vmap(net, in_axes=(None, 1), out_axes=0)(x, y)


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend910b'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_augassign_vmap_control_flow():
    """
    Feature: Support augassign inplace vmap in control flow.
    Description: Support augassign inplace vmap in control flow.
    Expectation: Run success.
    """
    net = ControlFlowNet()
    net.construct = jit(net.construct, backend='ms_backend')

    x = Tensor([0], dtype.float32)
    y = Tensor(np.ones([3, 4]), dtype.float32)
    vmap(net, in_axes=(None, 1), out_axes=0)(x, y)

    context.set_context(mode=context.GRAPH_MODE)
    vmap_net = VmapControlFlowNet()
    vmap_net(x, y)
