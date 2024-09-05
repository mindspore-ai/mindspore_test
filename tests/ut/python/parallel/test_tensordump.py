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


import numpy as np
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore import context
from mindspore.common.api import _cell_graph_executor
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

grad_all = C.GradOperation(get_all=True)

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x, y, b):
        predict = self.network(x, y, b)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, x, y, b):
        return grad_all(self.network)(x, y, b)

def compile_net(net, x, y, b1):
    net.set_train()
    _cell_graph_executor.compile(net, x, y, b1)


def test_tensordump_out_at_parameter():
    """
    Feature: test tensordump for construct parameter
    Description: x, y, b are both type of Tensor, the tensordump mode is 'out'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            ops.tensordump('input_x', x, 'out')
            ops.tensordump('input_y', y, 'out')
            ops.tensordump('input_b', b, 'out')
            out1 = self.matmul1(x, y)
            out2 = self.matmul2(out1, b)
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_tensordump_in_at_parameter():
    """
    Feature: test tensordump for construct parameter
    Description: x, y, b are both type of Tensor, the tensordump mode is 'in'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            ops.tensordump('input_x', x, 'in')
            ops.tensordump('input_y', y, 'in')
            ops.tensordump('input_b', b, 'in')
            out1 = self.matmul1(x, y)
            out2 = self.matmul2(out1, b)
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_tensordump_all_at_parameter():
    """
    Feature: test tensordump for construct parameter
    Description: x, y, b are both type of Tensor, the tensordump mode is 'all'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            ops.tensordump('input_x', x, 'all')
            ops.tensordump('input_y', y, 'all')
            ops.tensordump('input_b', b, 'all')
            out1 = self.matmul1(x, y)
            out2 = self.matmul2(out1, b)
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_tensordump_out_at_result():
    """
    Feature: test tensordump for construct result
    Description: out2 is type of Tensor, the tensordump mode is 'out'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            out2 = self.matmul2(out1, b)
            ops.tensordump('result', out2, 'out')
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_tensordump_in_at_result():
    """
    Feature: test tensordump for construct result
    Description: out2 is type of Tensor, the tensordump mode is 'in'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            out2 = self.matmul2(out1, b)
            ops.tensordump('result', out2, 'in')
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_tensordump_all_at_result():
    """
    Feature: test tensordump for construct result
    Description: out2 is type of Tensor, the tensordump mode is 'all'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            out2 = self.matmul2(out1, b)
            ops.tensordump('result', out2, 'all')
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_tensordump_out_between_ops():
    """
    Feature: test tensordump between two matmul,
    test tensordump op behavior under insertion of redistribution ops
    Description: tensordump mode is 'out'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            ops.tensordump('mul1_mul2', out1, 'out')
            out2 = self.matmul2(out1, b)
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_tensordump_in_between_ops():
    """
    Feature: test tensordump between two matmul,
    test tensordump op behavior under insertion of redistribution ops
    Description: tensordump mode is 'in'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            ops.tensordump('mul1_mul2', out1, 'in')
            out2 = self.matmul2(out1, b)
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_tensordump_all_between_ops():
    """
    Feature: test tensordump between two matmul,
    test tensordump op behavior under insertion of redistribution ops
    Description: tensordump mode is 'all'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            ops.tensordump('mul1_mul2', out1, 'all')
            out2 = self.matmul2(out1, b)
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)

def test_multiple_output():
    """
    Feature: test tensordump between two matmul,
    test tensordump op behavior under insertion of redistribution ops
    Description: out1 is used in multiple operators, tensordump mode is 'all'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.add = P.Add()

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            ops.tensordump('multi_output', out1, 'all')
            out2 = self.matmul2(out1, b)
            out3 = self.matmul3(out1, b)
            out4 = self.add(out2, out3)
            return out4

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    strategy3 = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3=strategy3)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)


def test_multiple_output_with_full_name():
    """
    Feature: test tensordump between two matmul,
    test tensordump op behavior under insertion of redistribution ops
    Description: out1 is used in multiple operators, tensordump mode is 'all'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.add = P.Add()

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            ops.tensordump('dumps/matmul1Result.npy', out1, 'all')
            out2 = self.matmul2(out1, b)
            out3 = self.matmul3(out1, b)
            out4 = self.add(out2, out3)
            return out4

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    strategy3 = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3=strategy3)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    compile_net(net, x, y, b)
