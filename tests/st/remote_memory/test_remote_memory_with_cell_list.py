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
import pytest
import numpy as np
from mindspore import mutable
from mindspore import jit, ops
from mindspore import Tensor, Parameter
from mindspore.nn import Cell, CellList
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cell_list_prefetch_and_detach():
    """
    Feature: Remote memory base operator
    Description: Base scene.
    Expectation: No Exception.
    """
    class NetA(Cell):
        def __init__(self):
            super(NetA, self).__init__()
            self.net_a_param_1 = Parameter(Tensor([1, 1, 1]), name="net_a_param_1")
            self.net_a_param_2 = Parameter(Tensor([2, 2, 2]), name="net_a_param_2")

        def construct(self, m):
            m = ops.relu(m + self.net_a_param_1 + self.net_a_param_2)
            return m


    class NetB(Cell):
        def __init__(self):
            super(NetB, self).__init__()
            self.net_b_param_1 = Parameter(Tensor([1, 1, 1]), name="net_b_param_1")
            self.net_b_param_2 = Parameter(Tensor([2, 2, 2]), name="net_b_param_2")

        def construct(self, m):
            m = ops.relu(m + self.net_b_param_1 + self.net_b_param_2)
            return m


    class NetC(Cell):
        def __init__(self):
            super(NetC, self).__init__()
            self.net_c_param_1 = Parameter(Tensor([1, 1, 1]), name="net_c_param_1")
            self.net_c_param_2 = Parameter(Tensor([2, 2, 2]), name="net_c_param_2")

        def construct(self, m):
            m = ops.relu(m + self.net_c_param_1 + self.net_c_param_2)
            return m


    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.cell_list = CellList()
            self.cell_list.append(NetA())
            self.cell_list.append(NetB())
            self.cell_list.append(NetC())
            self.layer_num = 3

        @jit
        def construct(self, m):
            for i in range(self.layer_num):
                m = self.cell_list[i](m)
            return m


    x = Tensor([1, 1, 1])
    net = Net()
    ret = net(x)
    assert np.all(ret.asnumpy() == np.array((10, 10, 10)))
