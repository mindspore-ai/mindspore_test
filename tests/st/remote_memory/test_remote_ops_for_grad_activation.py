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
import os
import pytest
import numpy as np
from mindspore import Tensor, Parameter, ops, jit
from mindspore.nn import Cell
from tests.mark_utils import arg_mark


@pytest.mark.skip(reason='wait for ops')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_grad_activation():
    """
    Feature: Test HyperMap insert remote memory ops.
    Description: HyperMap with remote memory ops.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    from mindspore import context
    context.set_context(save_graphs=True, save_graphs_path="./ir")

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_1 = Parameter(Tensor([[1, 1], [2, 2]]), name="param_1")
            self.param_2 = Parameter(Tensor([[3, 3], [4, 4]]), name="param_2")
            self.relu = ops.ReLU()

        def construct(self, a, b):
            x = ops.matmul(a, b)
            x_1 = x + self.param_1
            y = self.relu(x_1) # @jit.enable_remote_memory
            y_1 = y - self.param_2
            z = self.relu(y_1)
            return z

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = self.net.trainable_params()

        @jit
        def construct(self, a, b):
            return ops.grad(self.net, grad_position=None, weights=self.weights)(a, b)

    a = Tensor([[1, 1], [2, 2]])
    b = Tensor([[3, 3], [4, 4]])
    grad_net = GradNet(Net())
    ret = grad_net(a, b)
    assert np.all(ret[0].asnumpy() == np.array([[1, 1], [1, 1]]))
    assert np.all(ret[1].asnumpy() == np.array([[-1, -1], [-1, -1]]))

    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@pytest.mark.skip(reason='pass annotation for functional ops')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_remote_grad_activation_with_functional_operation():
    """
    Feature: Test HyperMap insert remote memory ops.
    Description: HyperMap with remote memory ops.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    from mindspore import context
    context.set_context(save_graphs=True, save_graphs_path="./ir")

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_1 = Parameter(Tensor([[1, 1], [2, 2]]), name="param_1")
            self.param_2 = Parameter(Tensor([[3, 3], [4, 4]]), name="param_2")

        def construct(self, a, b):
            x = ops.matmul(a, b)
            x_1 = x + self.param_1
            y = ops.relu(x_1) # @jit.enable_remote_memory
            y_1 = y - self.param_2
            z = ops.relu(y_1)
            return z

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = self.net.trainable_params()

        @jit
        def construct(self, a, b):
            return ops.grad(self.net, grad_position=None, weights=self.weights)(a, b)

    a = Tensor([[1, 1], [2, 2]])
    b = Tensor([[3, 3], [4, 4]])
    grad_net = GradNet(Net())
    ret = grad_net(a, b)
    assert np.all(ret[0].asnumpy() == np.array([[1, 1], [1, 1]]))
    assert np.all(ret[1].asnumpy() == np.array([[-1, -1], [-1, -1]]))

    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'


@pytest.mark.skip(reason='wait for ops')
@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_grad_result_to_remote():
    """
    Feature: Test HyperMap insert remote memory ops.
    Description: HyperMap with remote memory ops.
    Expectation: No Exception.
    """
    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '1'

    from mindspore import context
    context.set_context(save_graphs=True, save_graphs_path="./ir")

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.param_1 = Parameter(Tensor([[1, 1], [2, 2]]), name="param_1")
            self.param_2 = Parameter(Tensor([[3, 3], [4, 4]]), name="param_2")
            self.param_1.enable_grad_offload()
            self.relu = ops.ReLU()

        def construct(self, a, b):
            x = ops.matmul(a, b)
            x_1 = x + self.param_1
            y = self.relu(x_1)
            y_1 = y - self.param_2
            z = self.relu(y_1)
            return z

    class GradNet(Cell):
        def __init__(self, net):
            super(GradNet, self).__init__()
            self.net = net
            self.weights = self.net.trainable_params()

        @jit
        def construct(self, a, b):
            return ops.grad(self.net, grad_position=None, weights=self.weights)(a, b)

    a = Tensor([[1, 1], [2, 2]])
    b = Tensor([[3, 3], [4, 4]])
    grad_net = GradNet(Net())
    ret = grad_net(a, b)
    assert np.all(ret[0].asnumpy() == np.array([[1, 1], [1, 1]]))
    assert np.all(ret[1].asnumpy() == np.array([[-1, -1], [-1, -1]]))

    os.environ['MS_DEV_ENABLE_REMOTE_MEMORY'] = '0'
