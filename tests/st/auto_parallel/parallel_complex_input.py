# Copyright 2024 Huawei Technologies Co., Ltd
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
from mindspore import nn, ops
import mindspore.communication as D

import os

os.environ['HCCL_IF_BASE_PORT'] = '30000'


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = ops.Mul()
        self.mul.shard(((2, 2, 2), (2, 2, 2)))

    def construct(self, x, y):
        return self.mul(x, y)


def test_graph_mode():
    '''
    Feature: Parallel Support for Complex64 input
    Description: graph mode
    Expectation: Run success
    '''
    print(f"env 'HCCL_IF_BASE_PORT' is {os.environ['HCCL_IF_BASE_PORT']}")

    ms.set_context(mode=ms.GRAPH_MODE)
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.SEMI_AUTO_PARALLEL, dataset_strategy="full_batch")
    # Parallel in the case of complex input only supports KernelByKernel mode by now. So we set 'jit_level' to 'O0'.
    ms.set_context(jit_level='O0')

    D.init()
    ms.set_seed(1)

    x_real = np.random.randn(4, 4, 4).astype(np.float32)
    x_imag = np.random.randn(4, 4, 4).astype(np.float32)
    print(f"x_real is:\n{x_real}")
    print(f"x_imag is:\n{x_imag}")

    y_real = np.random.randn(4, 4, 4).astype(np.float32)
    y_imag = np.random.randn(4, 4, 4).astype(np.float32)
    print(f"y_real is:\n{y_real}")
    print(f"y_imag is:\n{y_imag}")

    x_np = x_real + 1j*x_imag
    y_np = y_real + 1j*y_imag

    x_ms = ms.Tensor(x_np)
    y_ms = ms.Tensor(y_np)

    output_np = x_np * y_np

    net = Net()
    output_ms = net(x_ms, y_ms)
    output_ms = output_ms.asnumpy()

    print(f"ms output real part is:\n{np.real(output_ms)}")
    print(f"ms output imag part is:\n{np.imag(output_ms)}")
    print(f"np output real part is:\n{np.real(output_np)}")
    print(f"np output imag part is:\n{np.imag(output_np)}")

    assert (np.allclose(np.real(output_np), np.real(output_ms), rtol=1e-03, atol=1e-03) and
            np.allclose(np.imag(output_np), np.imag(output_ms), rtol=1e-03, atol=1e-03)), \
            f"parallel complex st run failed, please check log_output."
    print(f"rank_{D.get_rank()} pass the test")
    ms.reset_auto_parallel_context()

test_graph_mode()
