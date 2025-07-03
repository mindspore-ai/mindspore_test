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
import mindspore as ms
from mindspore import Tensor, nn, ops

ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")

class GradNetWithFirstInput(nn.Cell):
    def __init__(self, net):
        super(GradNetWithFirstInput, self).__init__()
        self.net = net
        self.grad_op = ops.GradOperation(sens_param=True)

    def construct(self, x, dout):
        grad_fn = self.grad_op(self.net)
        return grad_fn(x, dout)


class Rsqrt(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = ops.rsqrt

    def construct(self, x):
        out = self.op(x)
        return out


def calc_grads_ms(inputs):
    net = Rsqrt()
    grad_net = GradNetWithFirstInput(net)
    grad_net.set_train()
    grads = []
    for x in inputs:
        out = net(*x)
        grad = grad_net(*x, out)
        grads.append(grad.asnumpy())
    return grads


def calc_grads_np(inputs):
    net = Rsqrt()
    grad_net = GradNetWithFirstInput(net)
    grad_net.set_train()
    grads = []
    for x in inputs:
        grad = np.power(x, -0.5) * np.power(x, -1.5) * (-0.5)
        grads.append(grad)
    return grads


if __name__ == '__main__':
    input_np1 = np.random.rand(5, 9, 8, 4, 5).astype(np.float32)
    inputs1 = [Tensor(input_np1)]
    input_np2 = np.random.rand(7, 4, 9, 5, 4, 9, 8).astype(np.float32)
    inputs2 = [Tensor(input_np1)]
    all_inputs = [inputs1, inputs2]

    gradients = calc_grads_ms(all_inputs)
    expects = calc_grads_np(all_inputs)

    assert np.allclose(gradients[0], expects[0])
    assert np.allclose(gradients[1], expects[1])
