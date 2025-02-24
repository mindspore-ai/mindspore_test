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
import os
import mindspore as ms
from mindspore.communication import init
from mindspore import nn
from mindspore import ops
from mindspore import Tensor


ms.set_context(mode=ms.GRAPH_MODE, device_target="Ascend")
ms.set_context(jit_level="O1")
np.random.seed(1)
init()

class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.matmul = ops.MatMul()
        self.allreduce = ops.AllReduce()

    def construct(self, x, y):
        z = self.matmul(x, y)
        z1 = self.allreduce(z)
        return z1

if __name__ == "__main__":
    tensor_a = Tensor(np.random.normal(0, 0.1, [1024, 2048]).astype(np.float16))
    tensor_b = Tensor(np.random.normal(0, 0.1, [2048, 1024]).astype(np.float16))
    net = Net()
    result = net(tensor_a, tensor_b).asnumpy()
    env_var = os.environ.get("MS_DEV_GRAPH_KERNEL_FLAGS")
    device_id = os.environ.get("DEVICE_ID")
    if env_var == "--enable_cluster_ops=MatMul,AllReduce":
        np.save("./dvm_matmul_allreduce_res_" + device_id + ".npy", result)
    else:
        np.save("./hccl_matmul_allreduce_res_" + device_id + ".npy", result)
