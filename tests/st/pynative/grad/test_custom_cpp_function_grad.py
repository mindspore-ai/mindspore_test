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
""" test_custom_cpp_function """

import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, Parameter, ops, nn
from mindspore.ops import CustomOpBuilder
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level2', card_mark='onecard', essential_mark='essential')
def test_custom_cpp_function_multi_input():
    """
    Feature: Custom cpp autograd function.
    Description: Custom forward function of multi input single output.
    Expectation: success.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.p = Parameter(2.0, requires_grad=True)
            self.my_ops = CustomOpBuilder("my_ops", ['./custom_src/function_ops.cpp'], backend="Ascend").load()

        def construct(self, x, y):
            z = self.my_ops.mul(x, y)
            return self.my_ops.mul(z, self.p)

    x = Tensor(1.0, ms.float32) * 2
    y = Tensor(1.0, ms.float32) * 3
    net = MyNet()
    grad_op = ms.value_and_grad(net, grad_position=(0, 1), weights=net.trainable_params())
    out, grads = grad_op(x, y)
    assert np.allclose(out.asnumpy(), np.array([12.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[0][0].asnumpy(), np.array([6.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[0][1].asnumpy(), np.array([4.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grads[1][0].asnumpy(), np.array([6.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level2', card_mark='onecard', essential_mark='essential')
def test_custom_cpp_function_mark_no_diff():
    """
    Feature: Custom cpp autograd function.
    Description: The output of function is marked non-diff.
    Expectation: success.
    """
    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.my_ops = CustomOpBuilder("my_ops", ['./custom_src/function_ops.cpp'], backend="Ascend").load()

        def construct(self, x, y):
            z = x * y
            return self.my_ops.mul_no_diff_out(z, y)

    x = Tensor([2, 2], ms.float32)
    y = Tensor([3, 3], ms.float32)
    grad_op = ops.GradOperation(get_all=True)
    grad = grad_op(MyNet())(x, y)
    assert np.allclose(grad[0].asnumpy(), np.array([0.0, 0.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grad[1].asnumpy(), np.array([0.0, 0.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level2', card_mark='onecard', essential_mark='essential')
def test_custom_cpp_function_mark_input_no_diff():
    """
    Feature: Custom cpp autograd function.
    Description: The output of function is marked non-diff when output is input.
    Expectation: success.
    """

    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.my_ops = CustomOpBuilder("my_ops", ['./custom_src/function_ops.cpp'], backend="Ascend").load()

        def construct(self, x):
            x_no_diff, _ = self.my_ops.mul_no_diff_in(x)
            return x * x_no_diff
    x = Tensor([2, 2], ms.float32)
    grad_op = ops.GradOperation(get_all=True)
    grad = grad_op(MyNet())(x)
    assert np.allclose(grad[0].asnumpy(), np.array([2.0, 2.0], dtype=np.float32), 0.00001, 0.00001)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level2', card_mark='onecard', essential_mark='essential')
def test_custom_cpp_function_dirty_tensor_is_need_grad_leaf():
    """
    Feature: Custom cpp autograd function.
    Description: The leaf tensor which needs grad is modified.
    Expectation: Raise RuntimeError.
    """

    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.my_ops = CustomOpBuilder("my_ops", ['./custom_src/function_ops.cpp'], backend="Ascend").load()

        def construct(self, x, y):
            return self.my_ops.inplace_mul(x, y)
    x = Tensor([2, 2], ms.float32)
    y = Tensor([3, 3], ms.float32)

    grad_op = ops.GradOperation(get_all=True)
    with pytest.raises(RuntimeError) as err:
        grad_op(MyNet())(x, y)
        assert "A leaf tensor that need grad is being used in an inplace operator" in str(err.value)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level2', card_mark='onecard', essential_mark='essential')
def test_custom_cpp_function_tensor_hook():
    """
    Feature: Custom cpp autograd function.
    Description: Verify the correctness of tensor hook in the context of custom cpp autograd function.
    Expectation: success.
    """

    class MyNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.my_ops = CustomOpBuilder("my_ops", ['./custom_src/function_ops.cpp'], backend="Ascend").load()

        def construct(self, x, y):
            z = self.my_ops.mul(x, y)
            def hook_fn(grad):
                return 2 * grad
            z.register_hook(hook_fn)
            return z

    x = Tensor([2, 2], ms.float32)
    y = Tensor([3, 3], ms.float32)
    grad_op = ops.GradOperation(get_all=True)
    grad = grad_op(MyNet())(x, y)
    assert np.allclose(grad[0].asnumpy(), np.array([6.0, 6.0], dtype=np.float32), 0.00001, 0.00001)
    assert np.allclose(grad[1].asnumpy(), np.array([4.0, 4.0], dtype=np.float32), 0.00001, 0.00001)
