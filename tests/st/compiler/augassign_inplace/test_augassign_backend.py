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

"""test augassign backend."""

import numpy as np
import mindspore as ms
from mindspore import nn
from mindspore import Tensor, jit
from mindspore import Parameter
from mindspore.common import dtype as mstype
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_augassign_backend():
    """
    Feature: Support augassign inplace in kbk mode.
    Description: Support augassign inplace in kbk mode.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.param_zero = Parameter(Tensor(0, mstype.float32), name='zero')
            self.param_a = Parameter(Tensor(15, mstype.float32), name='a')

        def construct(self):
            out0 = self.param_zero
            out1 = self.param_a

            out1 += self.param_a
            out0 += self.param_a
            return out0, out1

    pynative_output = Net()()

    net0 = Net()
    net0.construct = jit(net0.construct, backend='GE')
    graph_output_ge = net0()

    net1 = Net()
    net1.construct = jit(net1.construct, backend='ms_backend')
    graph_output = net1()

    assert graph_output_ge == pynative_output
    assert graph_output == pynative_output


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_initial_scalar_body_tensor1():
    """
    Feature: While specialize.
    Description: Test scalar arg when first entry of while and set to tensor in body.
    Expectation: No exception in infer process.
    """

    def func(x, a, b):
        y = 1
        while a < b:
            while a < b - 1:
                y = Tensor(2, ms.float32)
                a += 1
            a += 1
        return x + y

    @jit(backend='ms_backend')
    def test_net(x, a, b):
        out = x
        while a < b:
            while a < b - 1:
                out = func(out, a, b)
                a += 1
            a += 1
        return out

    input_np_x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    input_me_x = Tensor(input_np_x)
    input_me_a = Tensor(2, ms.float32)
    input_me_b = Tensor(6, ms.float32)
    test_net(input_me_x, input_me_a, input_me_b)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux', 'platform_ascend'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_fallback_assign_validation():
    """
    Feature: Support augassign inplace fallback with different input types
    Description: Test augassign inplace fallback with different input types
    Expectation: Run success.
    """

    @jit(backend='GE')
    def inplace_add_ext(x, y):
        x += Tensor(y)
        return x

    @jit(backend='GE')
    def inplace_adds_ext(x, y):
        x += y
        return x

    @jit(backend='GE')
    def inplace_sub_ext(x, y):
        x -= Tensor(y)
        return x

    @jit(backend='GE')
    def inplace_sub_scalar(x, y):
        x -= y
        return x

    @jit(backend='GE')
    def inplace_mul(x, y):
        x *= Tensor(y)
        return x

    @jit(backend='GE')
    def inplace_muls(x, y):
        x *= y
        return x

    @jit(backend='GE')
    def inplace_div(x, y):
        x /= Tensor(y)
        return x

    @jit(backend='GE')
    def inplace_divs(x, y):
        x /= y
        return x

    @jit(backend='GE')
    def inplace_floor_divide(x, y):
        x //= Tensor(y)
        return x

    @jit(backend='GE')
    def inplace_floor_divides(x, y):
        x //= y
        return x

    @jit(backend='GE')
    def inplace_remainder_tensor_tensor(x, y):
        x %= Tensor(y)
        return x

    @jit(backend='GE')
    def inplace_remainder_tensor_scalar(x, y):
        x %= y
        return x

    def test_assign_validation(f):
        input_y = 2.5
        input_x = Tensor(1)
        input_x_dtype = input_x.dtype
        output = f(input_x, input_y)
        assert input_x_dtype == output.dtype

    test_assign_validation(inplace_add_ext)
    test_assign_validation(inplace_adds_ext)
    test_assign_validation(inplace_sub_ext)
    test_assign_validation(inplace_sub_scalar)
    test_assign_validation(inplace_mul)
    test_assign_validation(inplace_muls)
    test_assign_validation(inplace_div)
    test_assign_validation(inplace_divs)
    test_assign_validation(inplace_floor_divide)
    test_assign_validation(inplace_floor_divides)
    test_assign_validation(inplace_remainder_tensor_tensor)
    test_assign_validation(inplace_remainder_tensor_scalar)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_validation():
    """
    Feature: Support augassign inplace grad with different input types
    Description: Fix the problem that the input types of AddN are not the same
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, y):
            y -= x
            x *= y
            return x, y

    x = Tensor(1)
    y = -2.5
    net = Net()
    net.construct = jit(net.construct, backend='GE')
    grad(net)(x, y)
