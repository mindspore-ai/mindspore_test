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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, nn, mutable
from mindspore.ops import ifft2


ms.context.set_context(jit_level="O0")

class IFFT2Net(nn.Cell):
    def __init__(self):
        super(IFFT2Net, self).__init__()
        self.ifft2 = ifft2

    def construct(self, x, s, dim):
        return self.ifft2(x, s, dim)


class IFFT2GradNet(nn.Cell):
    def __init__(self, net, dout):
        super(IFFT2GradNet, self).__init__()
        self.net = net
        self.dout = dout
        self.grad = ops.GradOperation(sens_param=True)

    def construct(self, x, s, dim):
        gout = self.grad(self.net)(x, s, dim, self.dout)
        return gout


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, s, dim):
    return np.fft.ifft2(x, s, dim)


def generate_expect_backward_output(x, s, dim):
    return np.fft.fft2(x, s, dim, norm="forward")


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ifft2_normal(mode):
    """
    Feature: ops.ifft2
    Description: test function ifft2 forward and backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    s = (2,)
    dim = (0,)
    net = IFFT2Net()
    output = net(ms.Tensor(x), s, dim)
    expect = generate_expect_forward_output(x, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    dout = generate_random_input((2, 3, 4, 5), np.complex64)
    grad_net = IFFT2GradNet(net, ms.Tensor(dout))
    grad_net.set_train()
    grad = grad_net(ms.Tensor(x), s, dim)
    expect = generate_expect_backward_output(dout, s, dim)
    np.testing.assert_allclose(grad.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ifft2_dynamic_shape(mode):
    """
    Feature: ops.ifft2
    Description: test function ifft2 forward and auto grad with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    s = (2,)
    dim = (0,)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)
    net = IFFT2Net()
    net.set_inputs(x_dyn, n_dyn, dim_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x1, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((2, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x2, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    ## auto grad
    net = IFFT2Net()
    s = (2,)
    dim = (0,)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    dout1 = generate_random_input((2, 3, 4, 5), np.complex64)
    grad_net = IFFT2GradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_backward_output(dout1, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((2, 4, 5, 6), np.float32)
    dout2 = generate_random_input((2, 4, 5, 6), np.complex64)
    grad_net = IFFT2GradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_backward_output(dout2, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_ifft2_dynamic_rank(mode):
    """
    Feature: ops.ifft2
    Description: test function ifft2 forward and auto grad with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    s = (2,)
    dim = (0,)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)
    net = IFFT2Net()
    net.set_inputs(x_dyn, n_dyn, dim_dyn)

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x1, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_forward_output(x2, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    ## auto grad
    s = (2,)
    dim = (0,)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    n_dyn = mutable(s)
    dim_dyn = mutable(dim)
    net = IFFT2Net()

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    dout1 = generate_random_input((2, 3, 4, 5), np.complex64)
    grad_net = IFFT2GradNet(net, ms.Tensor(dout1))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x1), n_dyn, dim_dyn)
    expect = generate_expect_backward_output(dout1, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)

    x2 = generate_random_input((2, 4, 5, 6), np.float32)
    dout2 = generate_random_input((2, 4, 5, 6), np.complex64)
    grad_net = IFFT2GradNet(net, ms.Tensor(dout2))
    grad_net.set_train()
    grad_net.set_inputs(x_dyn, n_dyn, dim_dyn)
    output = grad_net(ms.Tensor(x2), n_dyn, dim_dyn)
    expect = generate_expect_backward_output(dout2, s, dim)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-5)
