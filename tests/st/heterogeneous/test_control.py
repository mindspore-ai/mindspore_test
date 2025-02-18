# Copyright 2020 Huawei Technologies Co., Ltd
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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", jit_config={"jit_level": "O0"})


class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.relu1 = P.ReLU()
        self.relu2 = P.ReLU()
        self.mul = P.Mul()
        self.depend = P.Depend()

    def construct(self, x, y):
        a = self.relu1(x)
        y = self.depend(y, a)
        b = self.relu2(y)
        c = self.mul(a, b)
        return c, a


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.relu1 = P.ReLU()
        self.relu2 = P.ReLU().set_device("CPU")
        self.mul = P.Mul()
        self.depend = P.Depend()

    def construct(self, x, y):
        a = self.relu1(x)
        y = self.depend(y, a)
        b = self.relu2(y)
        c = self.mul(a, b)
        return c, a


@arg_mark(plat_marks=["platform_ascend"], level_mark="level1", card_mark="onecard", essential_mark="unessential")
def test_net():
    x = np.random.randn(2, 3, 3, 4).astype(np.float32)
    y = np.random.randn(2, 3, 3, 4).astype(np.float32)
    net1 = Net1()
    output1 = net1(Tensor(x), Tensor(y))

    net2 = Net2()
    output2 = net2(Tensor(x), Tensor(y))
    assert np.allclose(output1[0].asnumpy(), output2[0].asnumpy())
    print("##success##")


if __name__ == "__main__":
    test_net()
