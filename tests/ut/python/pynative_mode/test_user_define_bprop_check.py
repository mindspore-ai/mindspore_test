# Copyright 2020 Huawei Technologies Co., Ltd
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
""" test implicit conversion """
import numpy as np
import pytest

from mindspore import Tensor, nn, context
from mindspore.ops import composite as C
from mindspore._extends.parse import compile_config


grad_all_with_sens = C.GradOperation(get_all=True, sens_param=True)


def test_user_define_bprop_check_ok():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.grad = Tensor(np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], dtype=np.float32))

        def construct(self, x):
            ret = x * 2
            return ret

        def bprop(self, x, out, dout):
            return (self.grad * 3,)

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, sens):
            return grad_all_with_sens(self.net)(x, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    compile_config.CHECK_BPROP = 1
    net = Net()
    grad_net = GradNet(net)
    ret = grad_net(x, sens)
    compile_config.CHECK_BPROP = ''


def test_user_define_bprop_check_shape():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.grad = Tensor(np.array([[1.1, 2.2], [2.0, 3.0]], dtype=np.float32))

        def construct(self, x):
            ret = x * 2
            return ret

        def bprop(self, x, out, dout):
            return (self.grad * 3,)

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, sens):
            return grad_all_with_sens(self.net)(x, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    compile_config.CHECK_BPROP = 1
    net = Net()
    grad_net = GradNet(net)
    with pytest.raises(ValueError) as ex:
        ret = grad_net(x, sens)
    compile_config.CHECK_BPROP = ''


def test_user_define_bprop_check_dtype():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.grad = Tensor(np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], dtype=np.float16))

        def construct(self, x):
            ret = x * 2
            return ret

        def bprop(self, x, out, dout):
            return (self.grad * 3,)

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, sens):
            return grad_all_with_sens(self.net)(x, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    compile_config.CHECK_BPROP = 1
    net = Net()
    grad_net = GradNet(net)
    with pytest.raises(TypeError) as ex:
        ret = grad_net(x, sens)
    compile_config.CHECK_BPROP = ''


def test_user_define_bprop_check_number():
    class Net(nn.Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.grad = Tensor(np.array([[1.1, 2.2, 3.3], [2.0, 3.0, 4.0]], dtype=np.float32))

        def construct(self, x, y):
            ret = x * 2 + y
            return ret

        def bprop(self, x, y, out, dout):
            return (dout,)

    class GradNet(nn.Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net

        def construct(self, x, y, sens):
            return grad_all_with_sens(self.net)(x, y, sens)

    x = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    y = Tensor(np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    sens = Tensor(np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]], dtype=np.float32))
    context.set_context(mode=context.PYNATIVE_MODE)
    compile_config.CHECK_BPROP = 1
    net = Net()
    grad_net = GradNet(net)
    with pytest.raises(TypeError) as ex:
        ret = grad_net(x, y, sens)
    assert "For user defined method 'bprop' of net" in str(ex.value)
    compile_config.CHECK_BPROP = ''
