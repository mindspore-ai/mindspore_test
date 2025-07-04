# Copyright 2021 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, data_format="NCHW"):
        super(Net, self).__init__()
        self.bias_add = P.BiasAdd(data_format)

    def construct(self, x, b):
        return self.bias_add(x, b)


def get_output(x, b, data_format, enable_graph_kernel):
    jit_level = "O1" if enable_graph_kernel else "O0"
    context.set_context(jit_level=jit_level)
    net = Net(data_format)
    output = net(x, b)
    return output


def run_bias_add(shape1, shape2, data_format, dtype):
    np.random.seed(0)
    x = Tensor(np.random.normal(0, 10, shape1).astype(dtype))
    b = Tensor(np.ones(shape2).astype(dtype))
    expect = get_output(x, b, data_format, False)
    output = get_output(x, b, data_format, True)

    expect_np = expect.asnumpy().copy()
    output_np = output.asnumpy().copy()

    assert np.allclose(expect_np, output_np, 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bias_add_gpu():
    """
    Feature: test graph kernel BiasAdd
    Description: run test case on GPU
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_bias_add((2, 3), (3,), "NCHW", np.float32)
    run_bias_add((2, 3, 4, 5), (3,), "NCHW", np.float32)
    run_bias_add((2, 3, 4, 5), (5,), "NHWC", np.float32)

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bias_add_ascend():
    """
    Feature: test graph kernel BiasAdd
    Description: run test case on Ascend
    Expectation: the result match with expect
    """
    context.set_context(mode=context.GRAPH_MODE)
    run_bias_add((2, 3), (3,), "NCHW", np.float32)
    run_bias_add((2, 3, 4, 5), (3,), "NCHW", np.float32)
    run_bias_add((2, 3, 4, 5), (5,), "NHWC", np.float32)
