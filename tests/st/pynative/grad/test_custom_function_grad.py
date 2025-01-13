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
""" test_custom_function """

import numpy as np
import pytest
import mindspore
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore import nn
from mindspore import ops
from mindspore import Function
from tests.mark_utils import arg_mark

oneslike = P.OnesLike()


class MultiInputFunctionNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x*x
        z = t+y
        return z

    @staticmethod
    def backward(ctx, z):
        return z*3, z*4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_input():
    """
    Feature: Custom autograd function.
    Description: Custom forward function of multi input single output.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputFunctionNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


class MutiInputFunctionErrorNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x*x
        z = t+y
        return z

    @staticmethod
    def backward(ctx, z):
        return z*3


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_input_grad_num_wrong():
    """
    Feature: Custom autograd function.
    Description: The size of backward return is not eq to the size of forward inputs.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MutiInputFunctionErrorNet.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "Function backward return a wrong number of gradients" in str(err.value)


class MultiInputMultiOutputFunctionNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x*x
        z = t+y
        return z, t

    @staticmethod
    def backward(ctx, z, t):
        return z*3, z*4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_input_multi_output():
    """
    Feature: Custom autograd function.
    Description: Custom forward function of multi input multi output.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputMultiOutputFunctionNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


class MultiInputMultiOutputStarArgsFunctionNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x*x
        z = t+y
        return z, t

    @staticmethod
    def backward(ctx, *args):
        return args[0]*3, args[1]*4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_input_multi_output_star_args():
    """
    Feature: Custom autograd function.
    Description: Custom forward function of multi input multi output, backward input is *args.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputMultiOutputStarArgsFunctionNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


class MultiInputMultiOutputNotTensorFunctionNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x*x
        z = t+y
        return z, 4, t

    @staticmethod
    def backward(ctx, *args):
        return args[0]*3, args[2]*4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_input_multi_output_not_tensor():
    """
    Feature: Custom autograd function.
    Description: Some output of backward function is not tensor.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputMultiOutputNotTensorFunctionNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([3], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)


class MultiInputMultiOutputFunctionErrorNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x*x
        z = t+y
        return z, t

    @staticmethod
    def backward(ctx, z):
        return z*3, z*4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_backward_input_num_wrong():
    """
    Feature: Custom autograd function.
    Description: The size of backward inputs is not equal to the size of forward outputs.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputMultiOutputFunctionErrorNet.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(TypeError):
        grad_net(net)(x, y)


class MultiInputMultiOutputFunctionError1Net(Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x*x
        z = t+y
        return z, t

    @staticmethod
    def backward(ctx, z, t):
        return z*3, z*4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_no_tensor_grad_should_be_none():
    """
    Feature: Custom autograd function.
    Description: When the input of forward is not tensor, the corresponding output of backward should be none.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = 3
    net = MultiInputMultiOutputFunctionError1Net.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "Input is not tensor, but gradient is not none" in str(err.value)


class MultiInputMultiOutputFunctionError2Net(Function):
    @staticmethod
    def forward(ctx, x, y):
        t = x*x
        z = t+y
        return z, t

    @staticmethod
    def backward(ctx, z, t):
        return z*3, 4


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_grad_should_be_none_or_tensor():
    """
    Feature: Custom autograd function.
    Description: The outputs of backward should be nont or tensor.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = MultiInputMultiOutputFunctionError2Net.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "Gradient should be none or tensor" in str(err.value)


class CustomFunctionContextNet(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.age = 7
        x2 = x*x
        y = x2+1
        ctx.save_for_backward(x, x2, y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, x2, y = ctx.to_save
        age = ctx.age
        assert np.allclose(x.asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)
        assert np.allclose(x2.asnumpy(), np.array([4], dtype=np.float32), 0.00001, 0.00001)
        assert np.allclose(y.asnumpy(), np.array([5], dtype=np.float32), 0.00001, 0.00001)
        assert age == 7, "age should be 7."
        return x


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_context():
    """
    Feature: Custom autograd function.
    Description: The context can bring information from forward function to backward function.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    net = CustomFunctionContextNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x)
    assert np.allclose(grads[0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionNeedGradNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x*x
        z = x2+1
        return z

    @staticmethod
    def backward(ctx, grad_output):
        need_grad = ctx.needs_input_grad
        assert len(need_grad) == 2, "number of need grad should be same as input size."
        assert need_grad[0], "first input need grad"
        assert not need_grad[1], "second input do not need grad"
        return grad_output, None


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_need_grad():
    """
    Feature: Custom autograd function.
    Description: User can get the information of whether the inputs of forward function need grad.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = 3
    net = CustomFunctionNeedGradNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionNeedGradForwardNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        need_grad = ctx.needs_input_grad
        assert len(need_grad) == 2, "number of need grad should be same as input size."
        assert need_grad[0], "first input need grad"
        assert not need_grad[1], "second input do not need grad"
        x2 = x*x
        z = x2+1
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_need_grad_forward():
    """
    Feature: Custom autograd function.
    Description: User can get the information of whether the inputs of forward function need grad.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = 3
    net = CustomFunctionNeedGradForwardNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionDirtyTensorError1Net(Function):
    @staticmethod
    def forward(ctx, x, y):
        x.add_(1)
        ctx.mark_dirty(x)
        z = x+y
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output*2


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_dirty_tensor_must_all_be_output():
    """
    Feature: Custom autograd function.
    Description: Throw runtime error when exit dirty tensor is not output of forward function.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionDirtyTensorError1Net.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "The dirty tensors must all be outputs of the forward function." in str(err.value)


class CustomFunctionDirtyTensorError2Net(Function):
    @staticmethod
    def forward(ctx, x, y):
        x.add_(1)
        ctx.mark_dirty(x)
        z = x+y
        return z, x

    @staticmethod
    def backward(ctx, grad_output, a):
        return grad_output, grad_output*2


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_dirty_tensor_is_need_grad_leaf():
    """
    Feature: Custom autograd function.
    Description: Throw runtime error when leaf tensor need grad is modified.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionDirtyTensorError2Net.apply
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "A leaf tensor that need grad is being used in an inplace operator" in str(err.value)


class CustomFunctionDirtyTensorNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        x.add_(1)
        ctx.mark_dirty(x)
        z = x+y
        return z, x

    @staticmethod
    def backward(ctx, grad_output, a):
        return grad_output, grad_output*2


class CustomFunctionLeafNet(nn.Cell):
    def construct(self, y):
        x = Tensor([2, 2], mindspore.float32)
        return CustomFunctionDirtyTensorNet.apply(x, y)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_dirty_tensor_not_leaf_no_grad():
    """
    Feature: Custom autograd function.
    Description: Input tensor of forward function is not leaf, no need to do grad, can be modified.
    Expectation: success.
    """
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionLeafNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(y)
    assert np.allclose(grads[0].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionLeafNet1(nn.Cell):
    def construct(self, x, y):
        x = x + 1
        return CustomFunctionDirtyTensorNet.apply(x, y)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_dirty_tensor_not_leaf_need_grad():
    """
    Feature: Custom autograd function.
    Description: Input tensor of forward function is not leaf, need to do grad, can be modified.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionLeafNet1()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([1], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([2], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionNotLeafNet(nn.Cell):
    def construct(self, y):
        x = Tensor([2., 2, 2], mindspore.float32)
        x1 = x[1:3]
        return CustomFunctionDirtyTensorNet.apply(x1, y)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_dirty_view_multi_output_error():
    """
    Feature: Custom autograd function.
    Description: The scene dirty tensor is view and forward function is multi-output is prohibited.
    Expectation: success.
    """
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionNotLeafNet()
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(y)
    assert "A view is one of output for multi output operator" in str(err.value)


class CustomFunctionOutIsInNet(Function):
    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 2


class CustomFunctionWrapOutIsInNet(nn.Cell):
    def construct(self, y):
        y = ops.split(y, 1)[1]
        return CustomFunctionOutIsInNet.apply(y)


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_output_is_input():
    """
    Feature: Custom autograd function.
    Description: The output of forward function is input.
    Expectation: success.
    """
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionWrapOutIsInNet()
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(y)
    assert np.allclose(grads[0].asnumpy(), np.array([0, 2.], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionNonDiffNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x * x
        z = x2 + 1
        ctx.mark_non_differentiable(z)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output * 2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_non_diff():
    """
    Feature: Custom autograd function.
    Description: The output of forward function is marked non-diff.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionNonDiffNet.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([0.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([0.], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionNonDiffNet1(Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.mark_non_differentiable(x)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output * 2


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_non_diff1():
    """
    Feature: Custom autograd function.
    Description: The output of forward function is marked non-diff when output is input.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionNonDiffNet1.apply
    grad_net = C.GradOperation(get_all=True)
    grads = grad_net(net)(x, y)
    assert np.allclose(grads[0].asnumpy(), np.array([0.], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1].asnumpy(), np.array([0.], dtype=np.float32), 0.00001, 0.00001)


class CustomFunctionViewNotInAndDirtyNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x * x
        z = x2 + 1
        ctx.save_for_backward(x, y)
        w = z[0:1]
        return w

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.to_save
        return ops.OnesLike()(x), ops.OnesLike()(y) * 2


class CustomFunctionViewNotInAndDirtyWrapNet(nn.Cell):
    def construct(self, x, y):
        out = CustomFunctionViewNotInAndDirtyNet.apply(x, y)
        out.add_(1)
        return out


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_view_not_in_and_dirty():
    """
    Feature: Custom autograd function.
    Description: The output of forward function is a view, and is modified subsequently.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionViewNotInAndDirtyWrapNet()
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "This view tensor is output of custom cell" in str(err.value)


class CustomFunctionMultiDiffOutputNet(Function):
    @staticmethod
    def forward(ctx, x, y):
        x2 = x * x
        z = x2 + 1
        ctx.save_for_backward(x, y)
        v1 = z[0:1]
        return v1, x2

    @staticmethod
    def backward(ctx, grad_output, grad_output1):
        x, y = ctx.to_save
        return ops.OnesLike()(x), ops.OnesLike()(y) * 2


class CustomFunctionMultiDiffOutputWrapNet(nn.Cell):
    def construct(self, x, y):
        out = CustomFunctionMultiDiffOutputNet.apply(x, y)
        out[0].add_(1)
        return out


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_custom_function_multi_diff_output():
    """
    Feature: Custom autograd function.
    Description: The output of multi-output forward function is a view, and is modified subsequently.
    Expectation: success.
    """
    x = Tensor([2, 2], mindspore.float32)
    y = Tensor([3, 3], mindspore.float32)
    net = CustomFunctionMultiDiffOutputWrapNet()
    grad_net = C.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_net(net)(x, y)
    assert "A view of base is being inplace modified" in str(err.value)
