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
import os
import pytest
import numpy as np
from mindspore import Tensor, Parameter, ops, context
from mindspore.nn import Cell
from tests.mark_utils import arg_mark

# pylint: disable=unused-variable


@pytest.mark.skip(reason='wait for ops')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_hyper_map_remote_memory():
    """
    Feature: Test HyperMap insert remote memory ops.
    Description: HyperMap with remote memory ops.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'
    context.set_context(mode=context.GRAPH_MODE)
    func = ops.MultitypeFuncGraph("func")
    func.set_enable_remote_memory(True)

    @func.register("Tensor", "Tensor")
    def tensor_func(x, y):
        return ops.add(x, y)

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_1 = Parameter(Tensor([1, 1, 1]), name="param_1")
            self.param_2 = Parameter(Tensor([2, 2, 2]), name="param_2")
            self.param_3 = Parameter(Tensor([3, 3, 3]), name="param_3")
            self.param_4 = Parameter(Tensor([4, 4, 4]), name="param_4")
            self.hyper_map = ops.HyperMap(func)

        def construct(self, a, b, c, d):
            prefetch_list = ((b, self.param_2), (c, self.param_3), (d, self.param_4), ())
            m = self.hyper_map((a, b, c, d), (self.param_1, self.param_2, self.param_3, self.param_4), prefetch_list)
            return m

    a = Tensor([1, 1, 1])
    b = Tensor([1, 1, 1])
    c = Tensor([1, 1, 1])
    d = Tensor([1, 1, 1])
    net = Net()
    ret = net(a, b, c, d)
    assert np.all(ret[0].asnumpy() == np.array((2, 2, 2)))
    assert np.all(ret[1].asnumpy() == np.array((3, 3, 3)))
    assert np.all(ret[2].asnumpy() == np.array((4, 4, 4)))
    assert np.all(ret[3].asnumpy() == np.array((5, 5, 5)))
    context.set_context(mode=context.PYNATIVE_MODE)
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'
