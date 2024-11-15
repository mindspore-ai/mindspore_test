# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
""" test_bprop """

import torch
import numpy as np
import pytest
from mindspore import Tensor, nn, mint, ops, _no_grad
from mindspore.common.api import _pynative_executor
from mindspore.ops.auto_generate.gen_ops_def import as_strided, transpose, broadcast_to
from tests.st.pynative.utils import GradOfAllInputs
from tests.mark_utils import arg_mark


class ViewCopyNet(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        view = as_strided(z, (2, 2), (3, 1))
        view.copy_(y)
        return view


def view_copy_backward(x, y):
    x = x * 2
    view = x.as_strided((2, 2), (3, 1))
    view.copy_(y)
    view.sum().backward()


class TensorSliceCopyNet(nn.Cell):
    def construct(self, x):
        z = x * 2
        z[1] = 3
        z = z * 2
        z[2:3] = 3
        z = z * 2
        return z


def tensor_slice_copy_net(x):
    z = x * 2
    z[1] = 3
    z = z * 2
    z[2:3] = 3
    z = z * 2
    z.sum().backward()


class TensorIndexCopyNet(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        z[None] = 2
        z = z * y
        z[...] = 1
        z = z * y
        return z


def tensor_index_copy_net(x, y):
    z = x * 2
    z[None] = 2
    z = z * y
    z[...] = 1
    z = z * y
    z.sum().backward()


class CommonViewOpCopyNet1(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        z = transpose(z, (1, 0))
        z.copy_(y)
        z = x * y
        return z


def common_view_op_copy_net1(x, y):
    z = x * 2
    z = z.transpose(1, 0)
    z.copy_(y)
    z = x * y
    z.sum().backward()


class CommonViewOpCopyNet2(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        z = z.view((3, 2))
        z.copy_(y)
        z = ops.matmul(x, y)
        return z


def common_view_op_copy_net2(x, y):
    z = x * 2
    z = z.view(3, 2)
    z.copy_(y)
    z = torch.matmul(x, y)
    z.sum().backward()


class CommonViewOpCopyNet3(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        z = broadcast_to(z, (3, 3))
        z.copy_(y)
        z = x * y
        return z


def common_view_op_copy_net3(x, y):
    z = x * 2
    z = z.broadcast_to(3, 3)
    z.copy_(y)
    z = x * y
    z.sum().backward()


class MultiViewCopyNet(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        view1 = as_strided(z, (2, 2), (3, 1))
        view2 = as_strided(view1, (2, 1), (2, 1))
        view3 = transpose(view2, (1, 0))
        view3.copy_(y)
        return view2


def multi_view_copy_backward(x, y):
    z = x * 2
    view1 = z.as_strided((2, 2), (3, 1))
    view2 = view1.as_strided((2, 1), (2, 1))
    view3 = view2.transpose(1, 0)
    view3.copy_(y)
    view2.sum().backward()


class MultiViewCopyNet2(nn.Cell):
    def construct(self, x, y):
        z = x * 2
        z[1] = 1
        view1 = as_strided(z, (2, 2), (3, 1))
        view2 = as_strided(view1, (2, 1), (2, 1))
        view2.copy_(y)
        return z


def multi_view_copy_backward2(x, y):
    z = x * 2
    z[1] = 1
    view1 = z.as_strided((2, 2), (3, 1))
    view2 = view1.as_strided((2, 1), (2, 1))
    view2.copy_(y)
    z.sum().backward()


class MultiViewCopyNet3(nn.Cell):
    def construct(self, x, y, value):
        z = x * y
        view1 = z[2:4, 2:4]
        view2 = z[:2, :2]
        view3 = z[1:3, 1:3]
        view1.copy_(value)
        view2.copy_(value)
        view3.copy_(value)
        res = view1 + view2 + view3
        return res


def multi_view_copy_backward3(x, y, value):
    z = x * y
    view1 = z[2:4, 2:4]
    view2 = z[:2, :2]
    view3 = z[1:3, 1:3]
    view1.copy_(value)
    view2.copy_(value)
    view3.copy_(value)
    res = view1 + view2 + view3
    res.sum().backward()


class MultiCopyViewNet(nn.Cell):
    def construct(self, x, y, z):
        x = x * 2
        view1 = as_strided(x, (2, 2), (3, 1))
        view1.copy_(y)
        view2 = as_strided(view1, (2, 1), (2, 1))
        view2.copy_(z)
        return x


def multi_copy_view_backward(x, y, z):
    x = x * 2
    view1 = x.as_strided((2, 2), (3, 1))
    view1.copy_(y)
    view2 = view1.as_strided((2, 1), (2, 1))
    view2.copy_(z)
    x.sum().backward()


class ConstantTensorCopyViewNet(nn.Cell):
    def construct(self, x):
        x = x * 2
        y = Tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        view = y[1]
        view.copy_(x)
        return view


def constant_tensor_copy_view_net(x):
    x = x * 2
    y = torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    view = y[1]
    view.copy_(x)
    view.sum().backward()


class AsStridedExpandNet(nn.Cell):
    def construct(self, x):
        y = as_strided(x, (3, 3), (1, 0))
        return y


def as_strided_expand(x):
    y = x.as_strided((3, 3), (1, 0))
    y.sum().backward()


class AsStridedOverlapNet(nn.Cell):
    def construct(self, x):
        y = as_strided(x, (3, 2), (3, 6))
        return y


def as_strided_overlap(x):
    y = x.as_strided((3, 2), (3, 6))
    y.sum().backward()


class AsStridedInputOverlapNet(nn.Cell):
    def construct(self, x):
        y = mint.broadcast_to(x, (3, 3))
        z = as_strided(y, (1, 1), (1, 1))
        return z


def as_strided_input_overlap(x):
    y = x.expand(3, 3)
    z = y.as_strided((1, 1), (1, 1))
    z.sum().backward()


class NoGradViewCopyNet(nn.Cell):
    def construct(self, x, y):
        with _no_grad():
            view = as_strided(x, (2, 2), (2, 1))
        view.copy_(y)
        return view


class MultiOutputViewCopyNet(nn.Cell):
    def construct(self, x, y):
        view = mint.split(x, 2)
        view[0].copy_(y)
        return view


class LeafViewCopyNet(nn.Cell):
    def construct(self, x, y):
        x.copy_(y)
        return x

class OverlapViewCopyNet(nn.Cell):
    def construct(self, x, y):
        x.copy_(y)
        return x


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_view_copy():
    """
    Feature: Test view inplace grad.
    Description: test view copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y1 = Tensor([[1.0, 1.0], [1.0, 1.0]])
    view_copy_net = ViewCopyNet()
    view_copy_net.set_inputs()
    grad_fn1 = GradOfAllInputs(view_copy_net, sens_param=False)
    grads1 = grad_fn1(x1, y1)
    x_torch1 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    view_copy_backward(x_torch1, y_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())
    np.testing.assert_almost_equal(grads1[1].asnumpy(), y_torch1.grad.numpy())

    x2 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y2 = Tensor([[1.0, 1.0]])
    sens = Tensor([[1.0], [1.0]])
    multi_view_net = MultiViewCopyNet()
    multi_view_net.set_inputs()
    grad_fn2 = GradOfAllInputs(multi_view_net, sens_param=True)
    grads2 = grad_fn2(x2, y2, sens)
    x_torch2 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch2 = torch.tensor([[1.0, 1.0]], requires_grad=True)
    multi_view_copy_backward(x_torch2, y_torch2)
    np.testing.assert_almost_equal(grads2[0].asnumpy(), x_torch2.grad.numpy())
    np.testing.assert_almost_equal(grads2[1].asnumpy(), y_torch2.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_tensor_slice_copy():
    """
    Feature: Test tensor slice inplace grad.
    Description: test tensor copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]])
    sens = Tensor([[1., 1., 1.], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    tensor_slice_copy = TensorSliceCopyNet()
    tensor_slice_copy.set_inputs()
    grad_fn1 = GradOfAllInputs(tensor_slice_copy, sens_param=True)
    grads1 = grad_fn1(x1, sens)
    x_torch1 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]], requires_grad=True)
    tensor_slice_copy_net(x_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())

    x2 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]])
    y2 = Tensor([[1., 3., 1.], [1.0, 1.0, 3.0], [2.0, 2.0, 3.0]])
    sens = Tensor([[1., 1., 1.], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    tensor_index_copy = TensorIndexCopyNet()
    tensor_index_copy.set_inputs()
    grad_fn2 = GradOfAllInputs(tensor_index_copy, sens_param=True)
    grads2 = grad_fn2(x2, y2, sens)
    x_torch2 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]], requires_grad=True)
    y_torch2 = torch.tensor([[1., 3., 1.], [1.0, 1.0, 3.0], [2.0, 2.0, 3.0]], requires_grad=True)
    tensor_index_copy_net(x_torch2, y_torch2)
    np.testing.assert_almost_equal(grads2[0].asnumpy(), x_torch2.grad.numpy())
    np.testing.assert_almost_equal(grads2[1].asnumpy(), y_torch2.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_common_view_copy():
    """
    Feature: Test common view inplace grad.
    Description: test tensor copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]])
    y1 = Tensor([[1., 1., 3.], [2.0, 1.0, 1.0], [1.0, 2.0, 3.0]])
    sens = Tensor([[1., 1., 1.], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    net1 = CommonViewOpCopyNet1()
    net1.set_inputs()
    grad_fn1 = GradOfAllInputs(net1, sens_param=True)
    grads1 = grad_fn1(x1, y1, sens)
    x_torch1 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0], [3.0, 3.0, 3.0]], requires_grad=True)
    y_torch1 = torch.tensor([[1., 1., 3.], [2.0, 1.0, 1.0], [1.0, 2.0, 3.0]], requires_grad=True)
    common_view_op_copy_net1(x_torch1, y_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())
    np.testing.assert_almost_equal(grads1[1].asnumpy(), y_torch1.grad.numpy())

    x2 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y2 = Tensor([[1., 1.], [2.0, 1.0], [1.0, 1.0]])
    sens = Tensor([[1., 1.], [1.0, 1.0]])
    net2 = CommonViewOpCopyNet2()
    net2.set_inputs()
    grad_fn2 = GradOfAllInputs(net2, sens_param=True)
    grads2 = grad_fn2(x2, y2, sens)
    x_torch2 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch2 = torch.tensor([[1., 1.], [2.0, 1.0], [1.0, 1.0]], requires_grad=True)
    common_view_op_copy_net2(x_torch2, y_torch2)
    np.testing.assert_almost_equal(grads2[0].asnumpy(), x_torch2.grad.numpy())
    np.testing.assert_almost_equal(grads2[1].asnumpy(), y_torch2.grad.numpy())

    # x3 = Tensor([1., 2., 3.])
    # y3 = Tensor([[1., 1., 2.], [2.0, 1.0, 1.0], [1.0, 1.0, 2.0]])
    # sens = Tensor([[1., 1., 1.], [1.0, 1.0, 1.0], [1., 1., 1.]])
    # net3 = CommonViewOpCopyNet3()
    # net3.set_inputs()
    # grad_fn2 = GradOfAllInputs(net3, sens_param=True)
    # grads3 = grad_fn2(x3, y3, sens)
    # x_torch3 = torch.tensor([1., 2., 3.], requires_grad=True)
    # y_torch3 = torch.tensor([[1., 1., 2.], [2.0, 1.0, 1.0], [1.0, 1.0, 2.0]], requires_grad=True)
    # common_view_op_copy_net3(x_torch3, y_torch3)
    # np.testing.assert_almost_equal(grads3[0].asnumpy(), x_torch3.grad.numpy())
    # np.testing.assert_almost_equal(grads3[1].asnumpy(), y_torch3.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_multi_view_copy():
    """
    Feature: Test view inplace grad.
    Description: test view copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y1 = Tensor([[1.0, 1.0], [1.0, 1.0]])
    sens = Tensor([[1.0, 1.0], [1.0, 1.0]])
    view_copy_net = ViewCopyNet()
    view_copy_net.set_inputs()
    grad_fn1 = GradOfAllInputs(view_copy_net, sens_param=True)
    grads1 = grad_fn1(x1, y1, sens)
    x_torch1 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    view_copy_backward(x_torch1, y_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())
    np.testing.assert_almost_equal(grads1[1].asnumpy(), y_torch1.grad.numpy())

    x2 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y2 = Tensor([1.0, 1.0])
    sens = Tensor([[1.0], [1.0]])
    multi_view_net = MultiViewCopyNet()
    multi_view_net.set_inputs()
    grad_fn2 = GradOfAllInputs(multi_view_net, sens_param=True)
    grads2 = grad_fn2(x2, y2, sens)
    x_torch2 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch2 = torch.tensor([1.0, 1.0], requires_grad=True)
    multi_view_copy_backward(x_torch2, y_torch2)
    np.testing.assert_almost_equal(grads2[0].asnumpy(), x_torch2.grad.numpy())
    np.testing.assert_almost_equal(grads2[1].asnumpy(), y_torch2.grad.numpy())

    x3 = Tensor([[1., 2., 3.], [2., 1., 3.]])
    y3 = Tensor([[1.0], [2.0]])
    sens = Tensor([[1., 1., 1.], [1., 1., 1.]])
    multi_view_net2 = MultiViewCopyNet2()
    multi_view_net2.set_inputs()
    grad_fn3 = GradOfAllInputs(multi_view_net2, sens_param=True)
    grads3 = grad_fn3(x3, y3, sens)
    x_torch3 = torch.tensor([[1., 2., 3.], [2., 1., 3.]], requires_grad=True)
    y_torch3 = torch.tensor([[1.0], [2.0]], requires_grad=True)
    multi_view_copy_backward2(x_torch3, y_torch3)
    np.testing.assert_almost_equal(grads3[0].asnumpy(), x_torch3.grad.numpy())
    np.testing.assert_almost_equal(grads3[1].asnumpy(), y_torch3.grad.numpy())

    x4 = Tensor([[1., 2., 3., 4., 4.],
                 [1., 2., 3., 4., 4.],
                 [1., 2., 3., 0., 0.],
                 [1., 2., 5., 4., 4.],
                 [1., 2., 2., 4., 4.]])
    y4 = Tensor([[1., 2., 3., 4., 4.],
                 [1., 2., 3., 4., 4.],
                 [1., 2., 3., 0., 0.],
                 [1., 2., 1., 4., 4.],
                 [1., 1., 1., 4., 4.]])
    z4 = Tensor([[1., 2.], [2., 1.]])
    sens = Tensor([[1., 1.], [1., 1.]])
    multi_view_net3 = MultiViewCopyNet3()
    multi_view_net3.set_inputs()
    grad_fn3 = GradOfAllInputs(multi_view_net3, sens_param=True)
    grads4 = grad_fn3(x4, y4, z4, sens)
    x_torch4 = torch.tensor([[1., 2., 3., 4., 4.], [1., 2., 3., 4., 4.],
                             [1., 2., 3., 0., 0.], [1., 2., 5., 4., 4.],
                             [1., 2., 2., 4., 4.]], requires_grad=True)
    y_torch4 = torch.tensor([[1., 2., 3., 4., 4.], [1., 2., 3., 4., 4.],
                             [1., 2., 3., 0., 0.], [1., 2., 1., 4., 4.],
                             [1., 1., 1., 4., 4.]], requires_grad=True)
    z_torch4 = torch.tensor([[1., 2.], [2., 1.]], requires_grad=True)
    multi_view_copy_backward3(x_torch4, y_torch4, z_torch4)
    np.testing.assert_almost_equal(grads4[0].asnumpy(), x_torch4.grad.numpy())
    np.testing.assert_almost_equal(grads4[1].asnumpy(), y_torch4.grad.numpy())
    np.testing.assert_almost_equal(grads4[2].asnumpy(), z_torch4.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_view_multi_copy():
    """
    Feature: Test view inplace grad.
    Description: test view copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2.0, 1.0, 3.0]])
    y1 = Tensor([[1.0, 1.0], [1.0, 1.0]])
    z1 = Tensor([[3.0], [3.0]])
    sens = Tensor([[1., 1., 1.], [1.0, 1.0, 1.0]])
    view_copy_net = MultiCopyViewNet()
    view_copy_net.set_inputs()
    grad_fn1 = GradOfAllInputs(view_copy_net, sens_param=True)
    grads1 = grad_fn1(x1, y1, z1, sens)
    x_torch1 = torch.tensor([[1., 2., 3.], [2.0, 1.0, 3.0]], requires_grad=True)
    y_torch1 = torch.tensor([[1.0, 1.0], [1.0, 1.0]], requires_grad=True)
    z_torch1 = torch.tensor([[3.0], [3.0]], requires_grad=True)
    multi_copy_view_backward(x_torch1, y_torch1, z_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())
    np.testing.assert_almost_equal(grads1[1].asnumpy(), y_torch1.grad.numpy())
    np.testing.assert_almost_equal(grads1[2].asnumpy(), z_torch1.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_no_grad_copy():
    """
    Feature: Test view inplace grad.
    Description: test view copy grad.
    Expectation: No exception.
    """
    x1 = Tensor([1., 2., 3.])
    sens = Tensor([1., 1., 1.])
    view_copy_net = ConstantTensorCopyViewNet()
    view_copy_net.set_inputs()
    grad_fn1 = GradOfAllInputs(view_copy_net, sens_param=True)
    grads1 = grad_fn1(x1, sens)
    x_torch1 = torch.tensor([1., 2., 3.], requires_grad=True)
    constant_tensor_copy_view_net(x_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_as_strided_overlap_grad():
    """
    Feature: Test as strided grad.
    Description: test as strided grad.
    Expectation: with valid exception.
    """
    x1 = Tensor([1., 2., 3.])
    sens = Tensor([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]])
    as_strided_net = AsStridedExpandNet()
    as_strided_net.set_inputs()
    grad_fn1 = GradOfAllInputs(as_strided_net, sens_param=True)
    grads1 = grad_fn1(x1, sens)
    x_torch1 = torch.tensor([1., 2., 3.], requires_grad=True)
    as_strided_expand(x_torch1)
    np.testing.assert_almost_equal(grads1[0].asnumpy(), x_torch1.grad.numpy())

    x_torch2 = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13, 14, 15], requires_grad=True)
    as_strided_overlap(x_torch2)
    x2 = Tensor([1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13, 14, 15])
    sens = Tensor([[1., 1.], [1., 1.], [1., 1.]])
    overlap_net = AsStridedOverlapNet()
    overlap_net.set_inputs()
    grad_fn2 = GradOfAllInputs(overlap_net, sens_param=True)
    grads2 = grad_fn2(x2, sens)
    np.testing.assert_almost_equal(grads2[0].asnumpy(), x_torch2.grad.numpy())

    x_torch3 = torch.tensor([[1.0], [4.0], [7.0]], requires_grad=True)
    as_strided_input_overlap(x_torch3)
    x3 = Tensor([[1.0], [4.0], [7.0]])
    sens = Tensor([[1.0]])
    inputoverlap_net = AsStridedInputOverlapNet()
    inputoverlap_net.set_inputs()
    grad_fn3 = GradOfAllInputs(inputoverlap_net, sens_param=True)
    grads3 = grad_fn3(x3, sens)
    np.testing.assert_almost_equal(grads3[0].asnumpy(), x_torch3.grad.numpy())


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_view_inplace_grad_check_exception():
    """
    Feature: Test view inplace valid.
    Description: test view inplace valid
    Expectation: No exception.
    """
    x1 = Tensor([[1., 2., 3.], [2., 2., 2.], [3., 3., 3.]])
    y1 = Tensor([[1., 2.], [2., 2.]])
    sens = Tensor([[1., 1.], [1., 1.]])
    no_grad_view_net = NoGradViewCopyNet()
    no_grad_view_net.set_inputs()
    grad_fn1 = GradOfAllInputs(no_grad_view_net, sens_param=True)
    with pytest.raises(RuntimeError) as err:
        grad_fn1(x1, y1, sens)
        _pynative_executor.sync()
    assert "which created in no_grad mode and inplace modified with grad mode enabled" in str(err.value)

    x2 = Tensor([[1., 2.], [2., 2.], [3., 3.], [1, 1.]])
    y2 = Tensor([[1., 2.], [2., 2.]])
    sens = Tensor([[1., 1.], [1., 1.]])
    multi_output_view_net = MultiOutputViewCopyNet()
    multi_output_view_net.set_inputs()
    grad_fn2 = GradOfAllInputs(multi_output_view_net, sens_param=True)
    with pytest.raises(RuntimeError) as err:
        grad_fn2(x2, y2, sens)
        _pynative_executor.sync()
    assert "This view is one of output for multi output operator" in str(err.value)

    x3 = Tensor([[2., 3.], [2., 2.], [3., 3.]])
    y3 = Tensor([[1., 2.], [2., 2.], [1, 1]])
    sens = Tensor([[1., 1.], [1., 1.], [1, 1]])
    leaf_view_copy = LeafViewCopyNet()
    leaf_view_copy.set_inputs()
    grad_fn3 = GradOfAllInputs(leaf_view_copy, sens_param=True)
    with pytest.raises(RuntimeError) as err:
        grad_fn3(x3, y3, sens)
        _pynative_executor.sync()
    assert "A leaf tensor that requires grad is being used in an inplace operator" in str(err.value)

    x4 = Tensor([[2.], [2.], [3.]])
    y4 = Tensor([[1., 2.], [2., 2.], [1, 1]])
    sens = Tensor([[1., 1.], [1., 1.], [1, 1]])
    z4 = mint.broadcast_to(x4, (3, 2))
    overlap_view_copy = OverlapViewCopyNet()
    with pytest.raises(RuntimeError) as err:
        overlap_view_copy.construct(z4, y4)
        _pynative_executor.sync()
    assert "This tensor has multi element reference to the same memory address" in str(err.value)
