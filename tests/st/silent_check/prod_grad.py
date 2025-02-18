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

    def construct(self, *inputs):
        grad_fn = self.grad_op(self.net)
        return grad_fn(*inputs)


class Prod(nn.Cell):
    def __init__(self):
        super().__init__()
        self.op = ops.prod

    def construct(self, x, axis, keep_dims, dtype):
        out = self.op(x, axis, keep_dims, dtype)
        return out


if __name__ == '__main__':
    input_x = Tensor(np.random.randn(5,), ms.float32)
    input_axis = -1
    input_keep_dims = False
    input_dtype = ms.float32
    prod_net = Prod()
    output = prod_net(input_x, input_axis, input_keep_dims, input_dtype)
    sens = np.random.randn(*list(output.shape))
    grad_net = GradNetWithFirstInput(prod_net)
    grad_net.set_train()
    grad = grad_net(input_x, input_axis, input_keep_dims, input_dtype, Tensor(sens))
    print(grad.asnumpy().shape)
