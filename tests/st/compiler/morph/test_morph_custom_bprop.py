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
# ==============================================================================
"""Test morph with custom bprop"""

import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, Parameter, ops
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_morph_custom_bprop_001():
    """
    Feature: Morph
    Description: Test morph with custom bprop
    Expectation: No exception.
    """
    ms.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
    def infer_dtype(*args):
        return args[0]

    def infer_shape(*args):
        return args[0]

    def fn(x, y):
        return x * y

    def bprop_fn(x, y, out, dout):
        return (dout * y, dout * x)

    class TestNet0Morph(nn.Cell):
        def __init__(self, bprop_fn=None):
            super().__init__()
            self.weight0 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")
            self.morph = ops.Morph(fn, infer_shape, infer_dtype, bprop_fn=bprop_fn)  # bprop_fn 可选

        def construct(self, x):
            y = x * self.weight0
            z = self.morph(y, x)
            out = z * self.weight1
            return out

    x = np.array([7.0, 8.0, 9.0])
    input_x = Tensor(x)
    net = TestNet0Morph(bprop_fn)
    out_forward = net(input_x)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    bwd_out = grad_net(input_x)

    class TestNet0(nn.Cell):
        def __init__(self):
            super().__init__()
            self.weight0 = Parameter(Tensor(np.array([1.0, 2.0, 3.0]), ms.float32), name="weight0")
            self.weight1 = Parameter(Tensor(np.array([4.0, 5.0, 6.0]), ms.float32), name="weight1")

        def construct(self, x):
            y = x * self.weight0
            z = y * x
            out = z * self.weight1
            return out

    net_1 = TestNet0()
    out_forward_1 = net_1(input_x)
    grad_net_1 = grad_op(net_1, net_1.trainable_params())
    bwd_out_1 = grad_net_1(input_x)

    assert np.allclose(out_forward.asnumpy(), out_forward_1.asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[0][0].asnumpy(), bwd_out_1[0][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[1][0].asnumpy(), bwd_out_1[1][0].asnumpy(), 0.0001, 0.0001)
    assert np.allclose(bwd_out[1][1].asnumpy(), bwd_out_1[1][1].asnumpy(), 0.0001, 0.0001)
