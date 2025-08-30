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
import subprocess
import shutil
import numpy as np
from mindspore.nn import Cell
from mindspore.common import Tensor, Parameter
from mindspore.common.lazy_inline import lazy_inline
import mindspore.ops.operations as P
from mindspore import ops, nn, jit, context, recompute
from tests.mark_utils import arg_mark


class Grad(Cell):
    def __init__(self, net):
        super(Grad, self).__init__()
        self.grad = ops.GradOperation()
        self.net = net

    def construct(self, x):
        grad_net = self.grad(self.net)
        return grad_net(x)

class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        self.relu = P.ReLU()
        self.batch_size = 32
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid',
                               weight_init="normal")
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0, has_bias=False, pad_mode='valid',
                               weight_init="normal")
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.fc1 = nn.Dense(400, 120, weight_init="normal", bias_init="zeros")
        self.fc2 = nn.Dense(120, 84, weight_init="normal", bias_init="zeros")
        self.fc3 = nn.Dense(84, 10, weight_init="normal", bias_init="zeros")

    def construct(self, input_x):
        output = self.conv1(input_x)
        output = self.relu(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)
        output = self.reshape(output, (self.batch_size, -1))
        output = self.fc1(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.relu(output)
        output = self.fc3(output)
        return output


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_recompute_block_recompute1():
    """
    Feature: Sub cell recompute under gradjit
    Description: LeNet block is set recompute by the cell recompute api.
    Expectation: Run successfully.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.block = LeNet()
            self.block.recompute()

        @jit
        def construct(self, x):
            out = self.block(x)
            out = ops.Abs()(out)
            return out

    save_graphs_path = "./test_recompute_under_gradjit1"
    context.set_context(save_graphs=True, save_graphs_path=save_graphs_path)

    x = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    net = Net()
    grad_net = Grad(net)
    grad_net(x)

    para = '= Conv2D(%'
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, os.path.join(save_graphs_path, "opt_backward_[0-9]*.ir"))],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "2"

    if os.path.exists(save_graphs_path):
        shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_recompute_block_recompute1_with_func_api():
    """
    Feature: Sub cell recompute under gradjit
    Description: LeNet block is set recompute by the cell recompute api.
    Expectation: Run successfully.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.block = LeNet()

        @jit
        def construct(self, x):
            out = recompute(self.block, x)
            out = ops.Abs()(out)
            return out

    save_graphs_path = "./test_recompute_block_recompute1_with_func_api"
    context.set_context(save_graphs=True, save_graphs_path=save_graphs_path)

    x = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    net = Net()
    grad_net = Grad(net)
    grad_net(x)

    para = '= Conv2D(%'
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, os.path.join(save_graphs_path, "opt_backward_[0-9]*.ir"))],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "2"

    if os.path.exists(save_graphs_path):
        shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_recompute_block_recompute2():
    """
    Feature: Top cell recompute under gradjit
    Description: LeNet block is set recompute by the cell recompute api.
    Expectation: Run successfully.
    """

    class Net(Cell):
        def __init__(self):
            super(Net, self).__init__()
            self.block = LeNet()
            self.recompute()

        @jit
        def construct(self, x):
            out = self.block(x)
            out = ops.Abs()(out)
            return out

    save_graphs_path = "./test_recompute_under_gradjit2"
    context.set_context(save_graphs=True, save_graphs_path=save_graphs_path)

    x = Tensor(np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01)
    net = Net()
    grad_net = Grad(net)
    grad_net(x)

    para = '= Conv2D(%'
    output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (para, os.path.join(save_graphs_path, "opt_backward_[0-9]*.ir"))],
        shell=True)
    out = str(output, 'utf-8').strip()
    assert out == "0"

    if os.path.exists(save_graphs_path):
        shutil.rmtree(save_graphs_path)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_recompute_lazy_inline():
    """
    Feature: Recompute with lazyinline
    Description: Recompute with lazyinline
    Expectation: Run successfully.
    """
    class Block1(nn.Cell):
        def __init__(self, weight_shape=(32, 32)):
            super(Block1, self).__init__()
            self.mul = P.MatMul()
            self.add = P.Add()
            self.relu = P.ReLU()
            self.mul_weight = Parameter(Tensor(np.ones(weight_shape).astype(np.float32)), name="mul_weight")

        def construct(self, x):
            out = self.mul(x, self.mul_weight)
            out = self.relu(out)
            out = self.add(out, out)
            return out

    class Net(nn.Cell):
        @lazy_inline
        def __init__(self):
            super(Net, self).__init__()
            self.blocks = nn.CellList()
            b = Block1()
            self.blocks.append(b)

        @jit(backend="ms_backend")
        def construct(self, x):
            out = x
            out = self.blocks[0](out)
            return out

    net1 = Net()
    net2 = Net()
    net1.blocks[0].recompute()
    x = Tensor(np.random.randint(low=0, high=64, size=(32, 32)).astype(np.float32))
    grad1 = ops.grad(net1)(x)
    grad2 = ops.grad(net2)(x)
    assert np.allclose(grad1, grad2)
