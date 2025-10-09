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
import mindspore.context as context
from mindspore import ops, nn, Tensor


class RangeNet(nn.Cell):
    def __init__(self):
        super(RangeNet, self).__init__()
        self.shape = ops.Shape()

    def construct(self, x):
        shape = self.shape(x)
        return ops.range(0, shape[0], shape[1], 1000000)


def run_simple_reshape_net():
    net = RangeNet()
    net.set_inputs(Tensor(shape=[None, None], dtype=ms.float32))
    x = Tensor(np.ones([10, 2]).astype(np.float32))
    output = net(x)
    print("COMPILE", output, "CACHE")
    print("COMPILE", output.asnumpy().shape, "CACHE")


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE)
    context.set_context(jit_config={"jit_level": "O0"})
    run_simple_reshape_net()
