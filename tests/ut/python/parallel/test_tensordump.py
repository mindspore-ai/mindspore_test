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
from mindspore import Parameter, Tensor, ops
from mindspore import context
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss
from parallel.utils.utils import ParallelValidator, compile_net


def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

grad_all = C.GradOperation(get_all=True)


def get_tensordump_node_num(validator):
    d = validator.graph_info_dict
    res = 0
    for _, nodes in d.items():
        for node, _ in nodes.items():
            if node.startswith('TensorDump'):
                res += 1
    return res

class MatMulCell(nn.Cell):
    def __init__(self, hidden_size_config=(128, 128), strategy1=((1, 1), (1, 1))):
        super(MatMulCell, self).__init__()
        self.in_feature, self.out_feature = hidden_size_config
        self.matmul = P.MatMul().shard(strategy1)
        self.params = Parameter(Tensor(np.random.randn(self.in_feature, self.out_feature), dtype=ms.float32))

    def construct(self, x):
        out = self.matmul(x, self.params)
        return out

class SoftmaxCell(nn.Cell):
    def __init__(self):
        super(SoftmaxCell, self).__init__()
        self.softmax = P.Softmax(1)

    def construct(self, x):
        out = self.softmax(x)
        return out
class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, *x):
        predict = self.network(*x)
        return self.loss(predict)


class GradWrap(nn.Cell):
    def __init__(self, network):
        super(GradWrap, self).__init__()
        self.network = network

    def construct(self, *x):
        return grad_all(self.network)(*x)


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

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 3


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

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 3


def test_tensordump_inout_at_parameter():
    """
    Feature: test tensordump for construct parameter
    Description: x, y, b are both type of Tensor, the tensordump mode has both 'in' and 'out'
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

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 6


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

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 1


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

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 1


def test_tensordump_inout_at_result():
    """
    Feature: test tensordump for construct result
    Description: out2 is type of Tensor, the tensordump mode has both 'in' and 'out'
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
            ops.tensordump('resultOutSlice', out2, 'out')
            ops.tensordump('resultInSlice', out2, 'in')
            return out2

    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 2


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

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 1


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

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 1


def test_tensordump_inout_between_ops():
    """
    Feature: test tensordump between two matmul,
    test tensordump op behavior under insertion of redistribution ops
    Description: tensordump mode has both 'in' and 'out'
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            ops.tensordump('dumps/out1OutSlice.npy', out1, 'out')
            ops.tensordump('dumps/out1InSlice.npy', out1, 'in')
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

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 2

def test_multiple_output():
    """
    Feature: test tensordump between two matmul,
    test tensordump op behavior under insertion of redistribution ops
    Description: out1 is used in multiple operators
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
            ops.tensordump('dumps/out1OutSlice.npy', out1, 'out')
            ops.tensordump('dumps/out1InSlice.npy', out1, 'in')
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

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 3


def test_multiple_output_with_full_name():
    """
    Feature: test tensordump between two matmul,
    test tensordump op behavior under insertion of redistribution ops
    Description: out1 is used in multiple operators
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
            ops.tensordump('dumps/out1OutSlice.npy', out1, 'out')
            ops.tensordump('dumps/out1InSlice.npy', out1, 'in')
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

    phase = compile_net(net, x, y, b)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 3

def test_cell_level_dump_in_multi_output():
    """
    Feature: test tensordump in cell_level.
    test tensordump op behavior under insertion of redistribution ops
    Description: out1 is used in multiple operators, tensordump mode is 'in'
    Expectation: compile success
    """
    class CellLevelTensorDumpNet(nn.Cell):
        def __init__(self, input_dim, hidden_size1, hidden_size2, strategies):
            super(CellLevelTensorDumpNet, self).__init__()
            self.input_dim = input_dim
            self.hz1 = hidden_size1
            self. hz2 = hidden_size2
            st1, st2, st3 = strategies
            self.matmul1 = MatMulCell((self.input_dim, self.hz1), st1)
            self.matmul2 = MatMulCell((self.hz1, self.hz2), st2)
            self.matmul3 = MatMulCell((self.hz1, self.hz2), st3)
            self.add = P.Add()

        def construct(self, x):
            x = self.matmul1(x)
            ops.tensordump("cell_level_dump.npy", x, 'in')
            out1 = self.matmul2(x)
            out2 = self.matmul3(x)
            result = self.add(out1, out2)
            return result

    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    strategy3 = ((2, 2), (2, 2))
    input_x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    strategy_list = [strategy1, strategy2, strategy3]
    net = GradWrap(NetWithLoss(CellLevelTensorDumpNet(input_x.shape[1], 32, 32, strategy_list)))
    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 1

def test_cell_level_dump_inout_no_redistribution_op_insert():
    """
    Feature: test tensordump in cell_level.
    test tensordump op behavior in scenario of no redistribution operators inserted
    Description: out1 is used in multiple operators, no redistribution operator inserted
    Expectation: compile success
    """
    class CellLevelTensorDumpNet(nn.Cell):
        def __init__(self, input_dim, hidden_size1, hidden_size2, strategies):
            super(CellLevelTensorDumpNet, self).__init__()
            self.input_dim = input_dim
            self.hz1 = hidden_size1
            self.hz2 = hidden_size2
            st1, st2, st3 = strategies
            self.matmul1 = MatMulCell((self.input_dim, self.hz1), st1)
            self.matmul2 = MatMulCell((self.hz1, self.hz2), st2)
            self.matmul3 = MatMulCell((self.hz1, self.hz2), st3)
            self.add = P.Add()

        def construct(self, x):
            x = self.matmul1(x)
            ops.tensordump("no_redistribution_dump1", x, 'in')
            ops.tensordump("no_redistribution_dump2", x, 'in')
            out1 = self.matmul2(x)
            out2 = self.matmul3(x)
            result = self.add(out1, out2)
            return result

    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((1, 1), (1, 1))
    strategy2 = ((1, 1), (1, 1))
    strategy3 = ((1, 1), (1, 1))
    input_x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    strategy_list = [strategy1, strategy2, strategy3]
    net = GradWrap(NetWithLoss(CellLevelTensorDumpNet(input_x.shape[1], 32, 32, strategy_list)))
    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 2


def test_cell_level_dump_in_with_certain_cell():
    """
    Feature: test tensordump only in SoftmaxCell.
    test tensordump op behavior in scenario of dump single cell input
    Description: Test dump SoftmaxCell input
    Expectation: compile success
    """
    class SoftmaxCellWrapper(nn.Cell):
        def __init__(self):
            super(SoftmaxCellWrapper, self).__init__()
            self.softmax = SoftmaxCell()
        def construct(self, x):
            ops.tensordump("softmax_input_dump.npy", x, 'in')
            out = self.softmax(x)
            return out

    class SoftmaxCellTensorDumpNet(nn.Cell):
        def __init__(self, input_dim, hidden_size1, hidden_size2, strategies):
            super(SoftmaxCellTensorDumpNet, self).__init__()
            self.input_dim = input_dim
            self.hz1 = hidden_size1
            self.hz2 = hidden_size2
            self.softmax = SoftmaxCellWrapper()
            st1, st2, st3 = strategies
            self.matmul1 = MatMulCell((self.input_dim, self.hz1), st1)
            self.matmul2 = MatMulCell((self.hz1, self.hz2), st2)
            self.matmul3 = MatMulCell((self.hz1, self.hz2), st3)
            self.add = P.Add()

        def construct(self, x):
            x = self.matmul1(x)
            sft = self.softmax(x)
            out1 = self.matmul2(x)
            result = self.add(sft, out1)
            result = self.matmul3(result)
            return result

    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((4, 2), (2, 1))
    strategy2 = ((2, 4), (4, 1))
    strategy3 = ((2, 2), (2, 2))
    input_x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    strategy_list = [strategy1, strategy2, strategy3]
    net = GradWrap(NetWithLoss(SoftmaxCellTensorDumpNet(input_x.shape[1], 32, 32, strategy_list)))
    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    assert tensordump_num == 1
