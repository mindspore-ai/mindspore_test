# Copyright 2022 Huawei Technologies Co., Ltd
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
import pytest
import mindspore as ms
from mindspore import nn, ops, Tensor, context
from mindspore.ops import GradOperation
from mindspore.common import ParameterTuple
from mindspore.common.api import _pynative_executor
from mindspore._c_expression import CreationType
from tests.mark_utils import arg_mark


def forward_pre_hook_fn_add(cell, inp):
    x = inp[0] + inp[0]
    return x


def forward_pre_hook_fn_mul(cell, inp):
    x = inp[0] * inp[0]
    return x


def forward_hook_fn_relu(cell, inp, outp):
    out = nn.ReLU()(outp)
    return out


def forward_hook_fn_add(cell, inp, outp):
    out = outp + outp
    return out


def backward_hook_fn(cell, grad_inp, grad_outp):
    return Tensor(np.ones([1]).astype(np.float32)), Tensor(np.ones([1]).astype(np.float32))


def backward_hook_fn2(cell, grad_inp, grad_outp):
    return Tensor(np.ones([1]).astype(np.float32) * 2), Tensor(np.ones([1]).astype(np.float32) * 3)


def backward_hook_fn3(cell, grad_inp, grad_outp):
    return Tensor(np.ones([1]).astype(np.float32) * 5), Tensor(np.ones([1]).astype(np.float32) * 6)


def backward_hook_fn4(cell, grad_inp, grad_outp):
    return (Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 10),)


def backward_hook_fn5(cell, grad_inp, grad_outp):
    print("cell.a ", cell.a)
    cell.a = 2
    return grad_inp[0] * 2


unpair_v = 1


def backward_hook_fn6(cell, grad_inp, grad_outp):
    global unpair_v
    unpair_v += 1
    return (Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 10),)


def backward_hook_fn7(cell, grad_inp, grad_outp):
    print("grad_inp", grad_inp)
    print("grad_outp", grad_outp)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.mul = nn.MatMul()
        self.handle = self.mul.register_backward_hook(backward_hook_fn)

    def construct(self, x, y):
        x = self.mul(x, y)
        x = x + x
        return x


class SingleNet(nn.Cell):
    def __init__(self):
        super(SingleNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.conv.a = 1
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class DictNet(nn.Cell):
    def __init__(self):
        super(DictNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = self.conv(x)
        y = self.bn(x)
        return {'res': y, 'tmp': x}


class TestDictNet(nn.Cell):
    def __init__(self):
        super(TestDictNet, self).__init__()
        self.dict_net = DictNet()
        self.dict_net.a = 2

    def construct(self, x):
        z = self.dict_net(x)
        return z['res']


class DictInputNet(nn.Cell):
    def __init__(self):
        super(DictInputNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, *args, **kwargs):
        x = self.conv(args[0])
        y = self.bn(x)
        return y


class CmpNet(nn.Cell):
    def __init__(self):
        super(CmpNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class CmpNetPreHook(nn.Cell):
    def __init__(self):
        super(CmpNetPreHook, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")

    def construct(self, x):
        x = x + x
        x = x * x
        x = self.conv(x)
        x = self.bn(x)
        return x


class CmpNetFWHook(nn.Cell):
    def __init__(self):
        super(CmpNetFWHook, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")
        self.bn = nn.BatchNorm2d(2, momentum=0.99, eps=0.00001, gamma_init="ones")
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x + x
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_backward_hook():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward hook function.
    Expectation: The calculation result is correct.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([1]).astype(np.float32))
    input_y = Tensor(np.ones([1]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    # case 1: register hook function in __init__ function.
    net = Net()
    grad = grad_op(net)(input_x, input_y)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), input_x.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), input_x.asnumpy(), 0.000001, 0.000001)
    # case 2: remove hook function by handle.
    net.handle.remove()
    net.handle.remove()
    grad = grad_op(net)(input_x, input_y)
    assert len(grad) == 2
    expect_grad = Tensor(np.ones([1]).astype(np.float32) * 2)
    assert np.allclose(grad[0].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grad.asnumpy(), 0.000001, 0.000001)
    # case 3: register hook function by handle
    net = Net()
    net.mul.register_backward_hook(backward_hook_fn2)
    handle3 = net.mul.register_backward_hook(backward_hook_fn3)
    grad = grad_op(net)(input_x, input_y)
    assert len(grad) == 2
    expect_gradx = Tensor(np.ones([1]).astype(np.float32) * 5)
    expect_grady = Tensor(np.ones([1]).astype(np.float32) * 6)
    assert np.allclose(grad[0].asnumpy(), expect_gradx.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grady.asnumpy(), 0.000001, 0.000001)
    # case 5: remove hook function by handle.
    handle3.remove()
    grad = grad_op(net)(input_x, input_y)
    assert len(grad) == 2
    expect_gradx = Tensor(np.ones([1]).astype(np.float32) * 2)
    expect_grady = Tensor(np.ones([1]).astype(np.float32) * 3)
    assert np.allclose(grad[0].asnumpy(), expect_gradx.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1].asnumpy(), expect_grady.asnumpy(), 0.000001, 0.000001)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_base_line():
    """
    Feature: PyNative hook function.
    Description: The base line case for PyNative hook function.
    Expectation: The calculation result is correct.
    """

    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_all=True, get_by_list=True, sens_param=False)
    # register pre forward hook.
    net = SingleNet()
    handle1 = net.conv.register_forward_pre_hook(forward_pre_hook_fn_add)
    handle2 = net.conv.register_forward_pre_hook(forward_pre_hook_fn_mul)
    out = net(input_x)
    cmp_net_pre_hook = CmpNetPreHook()
    expect_out = cmp_net_pre_hook(input_x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    expect_grad = grad_op(cmp_net_pre_hook, ParameterTuple(cmp_net_pre_hook.trainable_params()))(input_x)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)
    # register forward hook.
    handle1.remove()
    handle2.remove()
    handlea = net.bn.register_forward_hook(forward_hook_fn_relu)
    handleb = net.bn.register_forward_hook(forward_hook_fn_add)
    out = net(input_x)
    cmp_net_fw_hook = CmpNetFWHook()
    expect_out = cmp_net_fw_hook(input_x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    expect_grad = grad_op(cmp_net_fw_hook, ParameterTuple(cmp_net_fw_hook.trainable_params()))(input_x)
    assert len(grad) == len(expect_grad)
    assert np.allclose(grad[0][0].asnumpy(), expect_grad[0][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)
    # register backward hook.
    handlea.remove()
    handleb.remove()
    net.conv.register_backward_hook(backward_hook_fn4)
    out = net(input_x)
    compare_net = CmpNet()
    expect_out = compare_net(input_x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), 0.000001, 0.000001)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    expect_grad = grad_op(compare_net, ParameterTuple(compare_net.trainable_params()))(input_x)
    assert len(grad) == len(expect_grad)
    expect_gradx = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 10)
    assert np.allclose(grad[0][0].asnumpy(), expect_gradx.asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][0].asnumpy(), expect_grad[1][0].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][1].asnumpy(), expect_grad[1][1].asnumpy(), 0.000001, 0.000001)
    assert np.allclose(grad[1][2].asnumpy(), expect_grad[1][2].asnumpy(), 0.000001, 0.000001)


class SpiltNet(nn.Cell):
    def construct(self, x, axis):
        return ops.split(x, axis)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_hook_tuple_with_single_element():
    """
    Feature: PyNative hook function.
    Description: The backward function input is a tuple with single element.
    Expectation: The calculation result is correct.
    """
    input_x = np.arange(9).astype("float32").reshape((1, 9))
    split_net = SpiltNet()
    split_net.register_backward_hook(backward_hook_fn)
    output = split_net(Tensor(input_x), 1)
    output_cat = ops.cat(output, axis=1)
    print(output_cat)


def test_pynative_backward_hook_with_modify_cell():
    """
    Feature: PyNative hook function.
    Description: Test PyNative backward hook function.
    Expectation: The calculation result is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_all=True, get_by_list=True, sens_param=False)
    # register backward hook.
    net = SingleNet()
    net.conv.register_backward_hook(backward_hook_fn5)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    assert len(grad) == 2
    assert net.conv.a == 2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_backward_hook_unpair():
    """
    Feature: PyNative backward hook function.
    Description: The unpair case for PyNative hook function.
    Expectation: The calculation result is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_by_list=True, sens_param=False)
    # register backward hook.
    net = SingleNet()
    net.conv.register_backward_hook(backward_hook_fn6)
    grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    assert unpair_v == 2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_backward_with_dict():
    """
    Feature: PyNative backward hook function.
    Description: The dict case for PyNative hook function.
    Expectation: The calculation result is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_by_list=True, sens_param=False)
    # register backward hook.
    net = TestDictNet()
    net.dict_net.register_backward_hook(backward_hook_fn6)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x)
    assert len(grad) == 3
    assert net.dict_net.a == 2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_backward_with_dict_input():
    """
    Feature: PyNative backward hook function.
    Description: The dict input case for PyNative hook function.
    Expectation: The calculation result is correct.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    input_y = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32) * 2)
    grad_op = GradOperation(get_by_list=True, sens_param=False)
    # register backward hook.
    net = DictInputNet()
    net.register_backward_hook(backward_hook_fn7)
    grad = grad_op(net, ParameterTuple(net.trainable_params()))(input_x, tmp=input_y)
    assert len(grad) == 3


class InplaceNet(nn.Cell):
    def __init__(self):
        super(InplaceNet, self).__init__()
        self.conv = nn.Conv2d(2, 2, kernel_size=2, stride=1, padding=0, weight_init="ones", pad_mode="valid")

    def construct(self, x, is_avoid_view_inplace_error):
        x = self.conv(x)
        if is_avoid_view_inplace_error:
            _pynative_executor.set_creation_type(x, CreationType.DEFAULT)
        x.add_(1.0)
        return x


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_backward_hook_single_inplace_net():
    """
    Feature: PyNative backward hook function.
    Description: Verify backward hook behavior when an inplace operation is applied to the single output of a Cell.
    Expectation: Pass when creation type is set to DEFAULT; Raise RuntimeError otherwise.
    """
    input_x = Tensor(np.ones([2, 2, 2, 2]).astype(np.float32))
    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    net = InplaceNet()
    net.conv.register_backward_hook(backward_hook_fn4)
    grad = grad_op(net)(input_x, True)
    assert len(grad) == 1
    assert np.allclose(grad[0].asnumpy(), np.ones([2, 2, 2, 2]).astype(np.float32) * 2, 0.000001)

    with pytest.raises(RuntimeError) as err:
        grad_op(net)(input_x, False)
        assert "This view tensor is output of custom cell, which has custom bprop" in err


class MultiInputInplaceNet(nn.Cell):
    def construct(self, x, y, is_avoid_view_inplace_error):
        if is_avoid_view_inplace_error:
            _pynative_executor.set_creation_type(x, CreationType.DEFAULT)
        x.mul_(2.0)
        return x * y


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_backward_hook_multi_inplace_net():
    """
    Feature: PyNative backward hook function.
    Description: Verify backward hook behavior when an inplace operation is applied to the multi input of a Cell.
    Expectation: Pass when creation type is set to DEFAULT; Raise RuntimeError otherwise.
    """
    input_x = Tensor([1.0, 1.0], dtype=ms.float32)
    input_y = Tensor([2.0, 3.0], dtype=ms.float32)

    def hook_fn(module, grad_in, grad_out):
        new_grad = []
        for grad_item in grad_in:
            if not grad_item is None:
                new_grad.append(grad_item * 2.0)
            else:
                new_grad.append(None)
        return tuple(new_grad)

    net = MultiInputInplaceNet()
    net.register_backward_hook(hook_fn)

    def fn(x, y, flag):
        x = x * 2
        return net(x, y, flag)

    grad_op = GradOperation(get_all=True, get_by_list=False, sens_param=False)
    grad = grad_op(fn)(input_x, input_y, True)
    assert len(grad) == 2
    assert np.allclose(grad[0].asnumpy(), np.array([8.0, 12.0], dtype=np.float32), 0.000001)
    assert np.allclose(grad[1].asnumpy(), np.array([8.0, 8.0], dtype=np.float32), 0.000001)

    with pytest.raises(RuntimeError) as err:
        grad_op(fn)(input_x, input_y, False)
        assert "This view is one of output for multi output operator" in err


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_backward_hook_pack_and_unpack():
    """
    Feature: PyNative backward hook function.
    Description: Pack and Unpack error-prone case for PyNative hook function.
    Expectation: The calculation result is correct.
    """
    class UnpackAndPackNet(nn.Cell):
        def construct(self, x):
            return x

    def backward_pre_hook(cell, grad_output):
        pass

    def backward_hook(cell, grad_input, grad_output):
        pass

    net = UnpackAndPackNet()
    net.register_backward_pre_hook(backward_pre_hook)
    net.register_backward_hook(backward_hook)

    input_x = (Tensor(1.0),)
    out = net(input_x)
    assert isinstance(out, tuple) and isinstance(out[0], Tensor)

    input_x = ((Tensor(1.0),),)
    out = net(input_x)
    assert isinstance(out, tuple) and isinstance(out[0], tuple) and isinstance(out[0][0], Tensor)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pynative_backward_recompute():
    """
    Feature: PyNative backward hook function.
    Description: Verify the correctness of backward hooks in recompute cell.
    Expectation: The calculation result is correct.
    """

    class RecomputeNet(nn.Cell):
        def construct(self, x):
            return x * x

    net = RecomputeNet()
    net.recompute()

    def double_hook(cell, grad_in, grad_out):
        return grad_in[0] * 2

    net.register_backward_hook(double_hook)
    x = Tensor([1.0, 2.0], dtype=ms.float32)
    grad_op = GradOperation(get_all=True)
    grad_x = grad_op(net)(x)
    assert np.allclose(grad_x[0].asnumpy(), np.array([4.0, 8.0], dtype=np.float32), 0.000001, 0.000001)
