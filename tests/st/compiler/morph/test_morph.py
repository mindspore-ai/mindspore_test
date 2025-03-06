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

import pytest
import numpy as np
import mindspore as ms
from mindspore import context, nn, ops, Tensor, Parameter
from tests.mark_utils import arg_mark

np_weight0 = np.array([1.0, 2.0, 3.0])
np_weight1 = np.array([4.0, 5.0, 6.0])
np_input_x = np.array([7.0, 8.0, 9.0])

def infer_dtype(*args):
    return args[0]

def infer_shape(*args):
    return args[0]

default_b = Tensor(2.0, ms.float32)
default_c = Tensor(3.0, ms.float32)
default_d = Tensor(4.0, ms.float32)

def mul_by(*args):
    def inner(a, b=default_b, c=default_c, d=default_d):
        x = args[0] * a
        x = x * default_b
        x = x * default_c
        x = x * default_d
        return x
    return inner

NUMBER_100 = 100

class TestMorphNet0(nn.Cell):
    def __init__(self):
        super(TestMorphNet0, self).__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")
        self.mul_by_100 = ops.Morph(mul_by(NUMBER_100), infer_shape, infer_dtype)

    def construct(self, x):
        o = x * self.weight0
        o = self.mul_by_100(o, default_b, default_c, default_d)
        out = o * self.weight1
        return out

class TestMorphNet1(nn.Cell):
    def __init__(self):
        super(TestMorphNet1, self).__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")
        self.mul_by_100 = ops.Morph(mul_by(NUMBER_100), infer_shape, infer_dtype)

    def construct(self, x):
        o = x * self.weight0
        o = self.mul_by_100(o, d=default_d, b=default_b)
        out = o * self.weight1
        return out

class TestMorphNet2(nn.Cell):
    def __init__(self):
        super(TestMorphNet2, self).__init__()
        self.weight0 = Parameter(Tensor(np_weight0, ms.float32), name="weight0")
        self.weight1 = Parameter(Tensor(np_weight1, ms.float32), name="weight1")
        self.mul_by_100 = ops.Morph(mul_by(NUMBER_100), infer_shape, infer_dtype)

    def construct(self, x):
        o = x * self.weight0
        o = self.mul_by_100(o, d=default_d, b=default_b)
        o = self.mul_by_100(o, c=default_c)
        o = self.mul_by_100(o)
        out = o * self.weight1
        return out

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("net, morph_call_time", [(TestMorphNet0(), 1), (TestMorphNet1(), 1), (TestMorphNet2(), 3)])
def test_morph_graph_mode(net, morph_call_time):
    """
    Feature: Morph Primitive
    Description: Test morph primitive for graph mode.
    Expectation: Run successfully.
    """
    context.set_context(mode=context.GRAPH_MODE)
    input_x = Tensor(np_input_x, ms.float32)
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    bwd_out = grad_net(input_x)
    x_grad = bwd_out[0][0].asnumpy()
    weight0_grad = bwd_out[1][0].asnumpy()
    weight1_grad = bwd_out[1][1].asnumpy()
    morph_dx = NUMBER_100 * default_b * default_c * default_d
    assert np.allclose(x_grad, np_weight1 * np_weight0 * morph_dx ** morph_call_time)
    assert np.allclose(weight0_grad, np_weight1 * np_input_x * morph_dx ** morph_call_time)
    assert np.allclose(weight1_grad, np_input_x * np_weight0 * morph_dx ** morph_call_time)

@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_morph_pynative_mode():
    """
    Feature: Morph Primitive
    Description: Test morph primitive for pynative mode.
    Expectation: Run successfully.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np_input_x, ms.float32)
    net = TestMorphNet0()
    grad_op = ops.GradOperation(get_all=True, get_by_list=True)
    grad_net = grad_op(net, net.trainable_params())
    with pytest.raises(RuntimeError) as e:
        grad_net(input_x)
    assert "Morph is only supported in GRAPH_MODE." in str(e.value)
