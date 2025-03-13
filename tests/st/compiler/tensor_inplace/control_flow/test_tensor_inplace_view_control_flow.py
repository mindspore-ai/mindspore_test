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

import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops.auto_generate.gen_ops_def import select_ext_view_op as select_ext_op
from mindspore.ops.auto_generate.gen_ops_def import expand_dims_view_op, slice_ext_view_op
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow1():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class NetPynative(nn.Cell):

        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = select_ext_op(input_tensor1, 0, 0)
            else:
                m = select_ext_op(input_tensor1, 0, 1)
            m.add_(x)
            return input_tensor1

    class NetJit(nn.Cell):

        @ms.jit
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = select_ext_op(input_tensor1, 0, 0)
            else:
                m = select_ext_op(input_tensor1, 0, 1)
            m.add_(x)
            return input_tensor1

    net = NetPynative()
    net_jit = NetJit()
    out_expect = grad(net)(Tensor(3), Tensor([1, 2]))
    out_jit = grad(net_jit)(Tensor(3), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow2():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class NetPynative(nn.Cell):

        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                if x > 2:
                    m = select_ext_op(input_tensor1, 0, 0)
                else:
                    m = select_ext_op(input_tensor1, 1, 0)
            else:
                if x < 7:
                    m = select_ext_op(input_tensor1, 0, 1)
                else:
                    m = select_ext_op(input_tensor1, 1, 1)
            m.add_(x)
            return input_tensor1

    class NetJit(nn.Cell):

        @ms.jit
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                if x > 2:
                    m = select_ext_op(input_tensor1, 0, 0)
                else:
                    m = select_ext_op(input_tensor1, 1, 0)
            else:
                if x < 7:
                    m = select_ext_op(input_tensor1, 0, 1)
                else:
                    m = select_ext_op(input_tensor1, 1, 1)
            m.add_(x)
            return input_tensor1

    net = NetPynative()
    net_jit = NetJit()
    out_expect = grad(net)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    out_jit = grad(net_jit)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow3():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class NetPynative(nn.Cell):

        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                if x > 2:
                    m = select_ext_op(input_tensor1, 0, 0)
                else:
                    m = select_ext_op(input_tensor1, 1, 0)
                m.sub_(x)
            else:
                if x < 7:
                    m = select_ext_op(input_tensor1, 0, 1)
                else:
                    m = select_ext_op(input_tensor1, 1, 1)
                m.add_(x)
            m.add_(x)
            return input_tensor1

    class NetJit(nn.Cell):

        @ms.jit
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                if x > 2:
                    m = select_ext_op(input_tensor1, 0, 0)
                else:
                    m = select_ext_op(input_tensor1, 1, 0)
                m.sub_(x)
            else:
                if x < 7:
                    m = select_ext_op(input_tensor1, 0, 1)
                else:
                    m = select_ext_op(input_tensor1, 1, 1)
                m.add_(x)
            m.add_(x)
            return input_tensor1

    net = NetPynative()
    net_jit = NetJit()
    out_expect = grad(net)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    out_jit = grad(net_jit)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow4():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class NetPynative(nn.Cell):

        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_op(input_tensor1, 0, 0)
            while m < 10:
                if x < 5:
                    n = select_ext_op(input_tensor1, 0, 1)
                    x = x + n
                else:
                    n = x
                m.add_(x)
            return input_tensor1

    class NetJit(nn.Cell):

        @ms.jit
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_op(input_tensor1, 0, 0)
            while m < 10:
                if x < 5:
                    n = select_ext_op(input_tensor1, 0, 1)
                    x = x + n
                else:
                    n = x
                m.add_(x)
            return input_tensor1

    net = NetPynative()
    net_jit = NetJit()
    out_expect = grad(net)(Tensor(0), Tensor([1, 2]))
    out_jit = grad(net_jit)(Tensor(0), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow5():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class NetPynative(nn.Cell):

        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = select_ext_op(input_tensor1, 0, 0)
            else:
                m = select_ext_op(input_tensor1, 0, 1)
            m.add_(x)
            if m > 5:
                n = select_ext_op(input_tensor1, 0, 1)
            else:
                n = select_ext_op(input_tensor1, 0, 0)
            n.add_(m)

            return input_tensor1

    class NetJit(nn.Cell):

        @ms.jit
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = select_ext_op(input_tensor1, 0, 0)
            else:
                m = select_ext_op(input_tensor1, 0, 1)
            m.add_(x)
            if m > 5:
                n = select_ext_op(input_tensor1, 0, 1)
            else:
                n = select_ext_op(input_tensor1, 0, 0)
            n.add_(m)

            return input_tensor1

    net = NetPynative()
    net_jit = NetJit()
    out_expect = grad(net)(Tensor(3), Tensor([1, 2]))
    out_jit = grad(net_jit)(Tensor(3), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow6():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class NetPynative(nn.Cell):

        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_op(input_tensor1, 0, 0)
            while m < 10:
                if x < 5:
                    n = select_ext_op(input_tensor1, 0, 1)
                    x = x + n
                else:
                    n = x
                m.add_(n)
                while n < 5:
                    n.add_(1)
            return input_tensor1

    class NetJit(nn.Cell):

        @ms.jit
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_op(input_tensor1, 0, 0)
            while m < 10:
                if x < 5:
                    n = select_ext_op(input_tensor1, 0, 1)
                    x = x + n
                else:
                    n = x
                m.add_(n)
                while n < 5:
                    n.add_(1)
            return input_tensor1

    net = NetPynative()
    net_jit = NetJit()
    out_expect = grad(net)(Tensor(3), Tensor([1, 2]))
    out_jit = grad(net_jit)(Tensor(3), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow7():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    def foo(x):
        m = select_ext_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_op(x, 0, 1)

    class NetPynative(nn.Cell):

        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = foo(input_tensor1)
            m.add_(x)
            return input_tensor1

    class NetJit(nn.Cell):

        @ms.jit
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = foo(input_tensor1)
            m.add_(x)
            return input_tensor1

    net = NetPynative()
    net_jit = NetJit()
    out_expect = grad(net)(Tensor(3), Tensor([1, 2]))
    out_jit = grad(net_jit)(Tensor(3), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow8():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    def foo(x):
        m = select_ext_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_op(x, 0, 1)

    class NetPynative(nn.Cell):

        def construct(self, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_op(input_tensor1, 0, 0)
            while m < 10:
                n = foo(input_tensor1)
                m.add_(n)
            return input_tensor1

    class NetJit(nn.Cell):

        @ms.jit
        def construct(self, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_op(input_tensor1, 0, 0)
            while m < 10:
                n = foo(input_tensor1)
                m.add_(n)
            return input_tensor1

    net = NetPynative()
    net_jit = NetJit()
    out_expect = grad(net)(Tensor([1, 2]))
    out_jit = grad(net_jit)(Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow9():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    def foo(x):
        m = select_ext_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_op(x, 0, 1)

    class NetPynative(nn.Cell):

        def construct(self, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_op(input_tensor1, 0, 0)
            while m < 10:
                n = foo(input_tensor1)
                m.add_(n)
            while m > 1:
                n = foo(input_tensor1)
                m.sub_(n)
            return input_tensor1

    class NetJit(nn.Cell):

        @ms.jit
        def construct(self, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_op(input_tensor1, 0, 0)
            while m < 10:
                n = foo(input_tensor1)
                m.add_(n)
            while m > 1:
                n = foo(input_tensor1)
                m.sub_(n)
            return input_tensor1

    net = NetPynative()
    net_jit = NetJit()
    out_expect = grad(net)(Tensor([1, 2]))
    out_jit = grad(net_jit)(Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow10():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class NetPynative(nn.Cell):

        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_op(input_tensor1_1, 0, 0)
            y = select_ext_op(input_tensor1_1, 0, 1)
            for _ in range(10):
                if x < 5:
                    for _ in range(5):
                        x.add_(y)
                        if x >= 5:
                            break
                    m = select_ext_op(input_tensor2_1, 0, 0)
                    n = select_ext_op(input_tensor2_1, 0, 1)
                else:
                    for _ in range(5):
                        x.sub_(y)
                        if x == 1:
                            continue
                    m = select_ext_op(input_tensor2_1, 0, 1)
                    n = select_ext_op(input_tensor2_1, 0, 0)
                m.add_(x)
                n.add_(y)
            return input_tensor2

    class NetJit(nn.Cell):

        @ms.jit
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_op(input_tensor1_1, 0, 0)
            y = select_ext_op(input_tensor1_1, 0, 1)
            for _ in range(10):
                if x < 5:
                    for _ in range(5):
                        x.add_(y)
                        if x >= 5:
                            break
                    m = select_ext_op(input_tensor2_1, 0, 0)
                    n = select_ext_op(input_tensor2_1, 0, 1)
                else:
                    for _ in range(5):
                        x.sub_(y)
                        if x == 1:
                            continue
                    m = select_ext_op(input_tensor2_1, 0, 1)
                    n = select_ext_op(input_tensor2_1, 0, 0)
                m.add_(x)
                n.add_(y)
            return input_tensor2

    net = NetPynative()
    net_jit = NetJit()
    out_expect = grad(net)(Tensor([3, 4]), Tensor([1, 2]))
    out_jit = grad(net_jit)(Tensor([3, 4]), Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow11():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    def foo(x):
        m = select_ext_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_op(x, 0, 1)

    class Net(nn.Cell):

        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_op(input_tensor1_1, 0, 0)
            y = select_ext_op(input_tensor1_1, 0, 1)
            for _ in range(10):
                if x < 5:
                    for _ in range(5):
                        x.add_(y)
                        if x >= 5:
                            break
                    m = foo(input_tensor2_1)
                    n = foo(input_tensor2_1)
                else:
                    for _ in range(5):
                        x.sub_(y)
                        if x == 1:
                            continue
                    m = foo(input_tensor2_1)
                    n = foo(input_tensor2_1)
                m.add_(x)
                n.add_(y)
            return input_tensor2

    net = Net()
    out_expect = grad(net)(Tensor([3, 4]), Tensor([1, 2]))
    net.construct = ms.jit(net.construct)
    out_jit = grad(net)(Tensor([3, 4]), Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow12():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    def foo(x):
        m = select_ext_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_op(x, 0, 1)

    class Net(nn.Cell):

        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            m = foo(input_tensor1_1)
            n = foo(input_tensor2_1)
            while m < 20:
                m.add_(n)
                for _ in range(3):
                    n.add_(m)
            return input_tensor2_1

    net = Net()
    out_expect = grad(net)(Tensor([1, 2]), Tensor([1, 2]))
    net.construct = ms.jit(net.construct)
    out_jit = grad(net)(Tensor([1, 2]), Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow13():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class Net(nn.Cell):

        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = expand_dims_view_op(input_tensor1, 0)
            else:
                m = expand_dims_view_op(input_tensor1, 1)
            m.add_(m)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor(0), Tensor([1, 2]))
    net.construct = ms.jit(net.construct)
    out_jit = grad(net)(Tensor(0), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow14():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class Net(nn.Cell):

        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = slice_ext_view_op(input_tensor1, 0, 0, 1, 1)
            else:
                m = slice_ext_view_op(input_tensor1, 0, 0, 1, 2)
            m.add_(m)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor(0), Tensor([1, 2]))
    net.construct = ms.jit(net.construct)
    out_jit = grad(net)(Tensor(0), Tensor([1, 2]))
    assert out_expect == out_jit
