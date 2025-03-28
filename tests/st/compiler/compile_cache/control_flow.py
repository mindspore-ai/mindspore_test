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
from mindspore.nn import Cell
import mindspore.context as context
from mindspore import Tensor, Parameter, ops, lazy_inline


def partial_without_inputs():
    class SubNet(Cell):
        @lazy_inline
        def __init__(self):
            super(SubNet, self).__init__()
            self.layer_norm = ops.LayerNorm()
            self.gamma = Tensor(np.ones([3]).astype(np.float32))
            self.beta = Tensor(np.ones([3]).astype(np.float32))

        def construct(self, x):
            output, _, _ = self.layer_norm(x, self.gamma, self.beta)
            return output

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.sub_net = SubNet()
            self.y = Parameter(Tensor(1))
            self.z = Parameter(Tensor(2))

        def construct(self, x):
            out = self.sub_net(x)
            if self.y > self.z:
                out = out * 2
            return out

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.grad = ops.GradOperation()
            self.net = net

        def construct(self, x):
            grad_func = self.grad(self.net)
            return grad_func(x)

    x = Tensor(np.array([[1, 2, 3], [1, 2, 3]]).astype(np.float32))
    grad_net = GradNet(Net())
    res = grad_net(x)
    print("AAA", res, "BBB")
    print("AAA", res.asnumpy().shape, "BBB")


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_level='O0')
    partial_without_inputs()
