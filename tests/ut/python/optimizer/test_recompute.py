# Copyright 2021 Huawei Technologies Co., Ltd
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
import mindspore.context as context
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore.common import Tensor, Parameter, recompute, jit
from mindspore import ops

recompute_prefix = 'recompute_'


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def construct(self, input_x):
        output = self.pool(input_x)
        return output


def test_set_recompute_true():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    net.pool.recompute()
    assert net.pool.get_scope() == recompute_prefix


def test_set_recompute_true_with_mp_comm_recompute():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    net.pool.recompute(mp_comm_recompute=True)
    assert net.pool.get_scope() == recompute_prefix


def test_set_recompute_true_with_mp_comm_recompute_false():
    context.set_context(mode=context.GRAPH_MODE)
    net = Net()
    net.pool.recompute(mp_comm_recompute=False)
    assert net.pool.get_scope() == recompute_prefix


class GradInJit(nn.Cell):
    def __init__(self, net):
        super(GradInJit, self).__init__()
        self.grad = ops.GradOperation()
        self.net = net

    @jit(backend="ms_backend")
    def construct(self, x):
        grad_net = self.grad(self.net)
        return grad_net(x)


class Block(nn.Cell):
    def __init__(self):
        super(Block, self).__init__()
        self.transpose1 = P.Transpose()
        self.transpose2 = P.Transpose()
        self.transpose3 = P.Transpose()
        self.transpose4 = P.Transpose()
        self.real_div1 = P.RealDiv()
        self.real_div2 = P.RealDiv()
        self.batch_matmul1 = P.BatchMatMul()
        self.batch_matmul2 = P.BatchMatMul()
        self.add = P.Add()
        self.softmax = P.Softmax(-1)
        self.dropout = P.Dropout(0.9)
        self.expand_dims = P.ExpandDims()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.y = Parameter(Tensor(np.ones((8, 128, 128)).astype(np.float32)))

    def construct(self, x):
        transpose1 = self.transpose1(x, (0, 2, 1, 3))
        real_div1 = self.real_div1(transpose1, Tensor(2.37891))
        transpose2 = self.transpose2(x, (0, 2, 3, 1))
        real_div2 = self.real_div2(transpose2, Tensor(2.37891))
        batch_matmul1 = self.batch_matmul1(real_div1, real_div2)
        expand_dims = self.expand_dims(self.y, 1)
        sub = self.sub(Tensor([1.0]), expand_dims)
        mul = self.mul(sub, Tensor([-0.0001]))
        add = self.add(mul, batch_matmul1)
        soft_max = self.softmax(add)
        dropout = self.dropout(soft_max)
        transpose3 = self.transpose3(x, (0, 2, 1, 3))
        batch_matmul2 = self.batch_matmul2(dropout[0], transpose3)
        transpose4 = self.transpose4(batch_matmul2, (0, 2, 1, 3))
        return transpose4


def test_recompute_nested_recompute_func_api1():
    """
    Feature: Recompute func api in jit.
    Description: Block is set recompute by the recomputed func api in the nested scene.
    Expectation: Raise an exception.
    """

    class OuterBlock(nn.Cell):
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()

        def construct(self, x):
            return self.block(x)

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                b.recompute()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = recompute(self.blocks[i], out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net1()
    grad_net = GradInJit(net)
    try:
        grad_net(x)
    except RuntimeError as e:
        assert "The cell passed into the recompute api should be set recomputed only once" in str(e)


def test_recompute_nested_recompute_func_api2():
    """
    Feature: Recompute func api in jit.
    Description: Block is set recompute by the recomputed func api in the nested scene.
    Expectation: Raise an exception.
    """

    class OuterBlock(nn.Cell):
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()

        def construct(self, x):
            return recompute(self.block, x)

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                b.recompute()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = self.blocks[i](out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net1()
    grad_net = GradInJit(net)
    try:
        grad_net(x)
    except RuntimeError as e:
        assert "The cell passed into the recompute api should be set recomputed only once" in str(e)


def test_recompute_nested_recompute_func_api3():
    """
    Feature: Recompute func api in jit.
    Description: Block is set recompute by the recomputed func api in the nested scene.
    Expectation: Raise an exception.
    """

    class OuterBlock(nn.Cell):
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()

        def construct(self, x):
            return recompute(self.block, x)

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = recompute(self.blocks[i], out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net1()
    grad_net = GradInJit(net)
    try:
        grad_net(x)
    except RuntimeError as e:
        assert "The cell passed into the recompute api should be set recomputed only once" in str(e)


def test_recompute_nested_recompute_func_api4():
    """
    Feature: Recompute func api in jit.
    Description: Block is set recompute by the recomputed func api in the nested scene.
    Expectation: Raise an exception.
    """

    class OuterBlock(nn.Cell):
        def __init__(self):
            super(OuterBlock, self).__init__()
            self.block = Block()
            self.block.recompute()

        def construct(self, x):
            return self.block(x)

    class Net1(nn.Cell):
        def __init__(self):
            super(Net1, self).__init__()
            self.blocks = nn.CellList()
            for _ in range(3):
                b = OuterBlock()
                self.blocks.append(b)

        def construct(self, x):
            out = x
            for i in range(3):
                out = recompute(self.blocks[i], out)
            return out

    x = Tensor(np.ones((8, 128, 16, 32)).astype(np.float32))
    net = Net1()
    grad_net = GradInJit(net)
    try:
        grad_net(x)
    except RuntimeError as e:
        assert "The cell passed into the recompute api should be set recomputed only once" in str(e)
