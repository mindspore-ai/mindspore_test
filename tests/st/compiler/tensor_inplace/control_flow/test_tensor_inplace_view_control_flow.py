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
from mindspore import Tensor, nn, ops, mint
from mindspore.ops.auto_generate.gen_ops_def import select_ext_view_op, expand_dims_view_op, slice_ext_view_op
from mindspore.ops.auto_generate.gen_ops_prim import inplace_copy_op
from mindspore.ops.functional import grad
from tests.mark_utils import arg_mark

ms.context.set_context(jit_config={"jit_level": "O0"})

@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_view_in_control_flow1():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = select_ext_view_op(input_tensor1, 0, 0)
            else:
                m = select_ext_view_op(input_tensor1, 0, 1)
            m.add_(x)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor(3), Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(3), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_view_in_control_flow2():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                if x > 2:
                    m = select_ext_view_op(input_tensor1, 0, 0)
                else:
                    m = select_ext_view_op(input_tensor1, 1, 0)
            else:
                if x < 7:
                    m = select_ext_view_op(input_tensor1, 0, 1)
                else:
                    m = select_ext_view_op(input_tensor1, 1, 1)
            m.add_(x)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_view_in_control_flow3():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                if x > 2:
                    m = select_ext_view_op(input_tensor1, 0, 0)
                else:
                    m = select_ext_view_op(input_tensor1, 1, 0)
                m.add_(x)
            else:
                if x < 7:
                    m = select_ext_view_op(input_tensor1, 0, 1)
                else:
                    m = select_ext_view_op(input_tensor1, 1, 1)
                m.add_(x)
            m.add_(x)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(3), Tensor([[1, 2], [3, 4]]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow4():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            while m < 10:
                if x < 5:
                    n = select_ext_view_op(input_tensor1, 0, 1)
                    x = x + n
                else:
                    n = x
                m.add_(x)
            return input_tensor1


    net = Net()
    out_expect = grad(net)(Tensor(0), Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(0), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow5():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            if x < 5:
                m = select_ext_view_op(input_tensor1, 0, 0)
            else:
                m = select_ext_view_op(input_tensor1, 0, 1)
            m.add_(x)
            if m > 5:
                n = select_ext_view_op(input_tensor1, 0, 1)
            else:
                n = select_ext_view_op(input_tensor1, 0, 0)
            n.add_(m)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor(3), Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(3), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow6():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            while m < 10:
                if x < 5:
                    n = select_ext_view_op(input_tensor1, 0, 1)
                    x = x + n
                else:
                    n = x
                m.add_(n)
                while n < 5:
                    n.add_(1)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor(3), Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(3), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_view_in_control_flow7():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    def foo(x):
        m = select_ext_view_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_view_op(x, 0, 1)

    class Net(nn.Cell):
        def construct(self, x, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = foo(input_tensor1)
            m.add_(x)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor(3), Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(3), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow8():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    def foo(x):
        m = select_ext_view_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_view_op(x, 0, 1)

    class Net(nn.Cell):
        def construct(self, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            while m < 10:
                n = foo(input_tensor1)
                m.add_(n)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow9():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    def foo(x):
        m = select_ext_view_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_view_op(x, 0, 1)

    class Net(nn.Cell):
        def construct(self, input_tensor):
            input_tensor1 = ops.abs(input_tensor)
            m = select_ext_view_op(input_tensor1, 0, 0)
            while m < 10:
                n = foo(input_tensor1)
                m.add_(n)
            while 1 < m < 5:
                n = foo(input_tensor1)
                m.add_(n)
            return input_tensor1

    net = Net()
    out_expect = grad(net)(Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow10():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor1_1, 0, 1)
            for _ in range(10):
                if x < 5:
                    for _ in range(5):
                        x.add_(y)
                        if x >= 5:
                            break
                    m = select_ext_view_op(input_tensor2_1, 0, 0)
                    n = select_ext_view_op(input_tensor2_1, 0, 1)
                else:
                    for _ in range(5):
                        x.add_(y)
                        if x == 1:
                            continue
                    m = select_ext_view_op(input_tensor2_1, 0, 1)
                    n = select_ext_view_op(input_tensor2_1, 0, 0)
                m.add_(x)
                n.add_(y)
            return input_tensor2

    net = Net()
    out_expect = grad(net)(Tensor([3, 4]), Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([3, 4]), Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow11():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    def foo(x):
        m = select_ext_view_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_view_op(x, 0, 1)

    class Net(nn.Cell):
        def construct(self, input_tensor1, input_tensor2):
            input_tensor1_1 = ops.abs(input_tensor1)
            input_tensor2_1 = ops.abs(input_tensor2)
            x = select_ext_view_op(input_tensor1_1, 0, 0)
            y = select_ext_view_op(input_tensor1_1, 0, 1)
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
                        x.add_(y)
                        if x == 1:
                            continue
                    m = foo(input_tensor2_1)
                    n = foo(input_tensor2_1)
                m.add_(x)
                n.add_(y)
            return input_tensor2_1

    net = Net()
    out_expect = grad(net)(Tensor([3, 4]), Tensor([1, 2]))
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([3, 4]), Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.skip(reason="No support")
def test_view_in_control_flow12():
    """
    Feature: view operation in control flow.
    Description: test view operation in control flow.
    Expectation: no exception
    """
    def foo(x):
        m = select_ext_view_op(x, 0, 0)
        if m > 5:
            return m
        return select_ext_view_op(x, 0, 1)

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
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor([1, 2]), Tensor([1, 2]))
    assert np.allclose(out_expect.asnumpy(), out_jit.asnumpy())


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(0), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(Tensor(0), Tensor([1, 2]))
    assert out_expect == out_jit


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_with_ctr_flow():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, a, b):
            y = ops.abs(x)
            if mint.equal(a, b):
                y_viewed = select_ext_view_op(y, 0, 0)
            else:
                y_viewed = select_ext_view_op(y, 1, 1)
            inplace_copy_op(y_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y

    x_np = (np.arange(2 * 2)).reshape((2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    net = Net()
    out_expect = grad(net)(x, x, x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x, x, x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_with_ctr_flow2():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, a, b):
            y = ops.abs(x)
            if mint.equal(a, b):
                y_viewed1 = select_ext_view_op(y, 0, 0)
                y_viewed2 = select_ext_view_op(y, 0, 1)
            else:
                y_viewed1 = select_ext_view_op(y, 1, 0)
                y_viewed2 = select_ext_view_op(y, 1, 1)
            inplace_copy_op(y_viewed1, ms.Tensor(-1, dtype=ms.float32))
            y_viewed2.add_(2)
            return y

    x_np = (np.arange(2 * 2)).reshape((2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    net = Net()
    out_expect = grad(net)(x, x, x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x, x, x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor_view_inplace_grad_with_ctr_flow3():
    """
    Feature: Support tensor inplace view gradient.
    Description: Support tensor inplace view gradient.
    Expectation: Run success.
    """

    class Net(nn.Cell):
        def construct(self, x, a, b):
            y = ops.abs(x)
            y_viewed = select_ext_view_op(y, 0, 0)
            if mint.equal(a, b):
                y_viewed.add_(-2)
            else:
                inplace_copy_op(y_viewed, ms.Tensor(-1, dtype=ms.float32))
            return y

    x_np = (np.arange(2 * 2)).reshape((2, 2)).astype(np.float32)
    x = ms.Tensor(x_np)
    net = Net()
    out_expect = grad(net)(x, x, x)
    net.construct = ms.jit(net.construct, backend="ms_backend")
    out_jit = grad(net)(x, x, x)
    assert (out_expect.asnumpy() == out_jit.asnumpy()).all()
