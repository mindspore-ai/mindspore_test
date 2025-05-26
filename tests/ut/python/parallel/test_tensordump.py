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

import os
import sys
import glob
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.log as logger
from mindspore import Parameter, Tensor, ops
from mindspore import context
from mindspore.ops import composite as C
from mindspore.ops import operations as P
from tests.ut.python.ops.test_math_ops import VirtualLoss
from mindspore.ops.auto_generate import GroupedMatmul  # pylint: disable=ungrouped-imports
from parallel.utils.utils import ParallelValidator, compile_net

def setup_function():
    context.set_auto_parallel_context(dataset_strategy="full_batch")

def gen_save_graph_path_by_testcase(testcase_name):
    abs_path = os.path.abspath(__file__)
    dir_name = os.path.dirname(abs_path)
    return f"{dir_name}/{testcase_name}"

grad_all = C.GradOperation(get_all=True)

def get_tensordump_node_num(validator):
    d = validator.graph_info_dict
    res = 0
    for _, nodes in d.items():
        for node, _ in nodes.items():
            if node.startswith('TensorDump'):
                res += 1
    return res

def get_tensordump_node_infos(graph_validator, reserve_node='TensorDump'):
    d = graph_validator.graph_info_dict
    tensordump_node_infos = []
    for _, nodes in d.items():
        for node, node_info in nodes.items():
            if node.startswith(reserve_node):
                tensordump_node_infos.append(node_info)
    return tensordump_node_infos

def check_dump_path_and_attr(info_list, expect_dump_path, expect_attr_dict):
    for td_info in info_list:
        # info format: {'inputs': [], 'attrs' {}}
        node_dump_path = td_info['inputs'][0]
        node_attr = td_info['attrs']
        if node_dump_path == expect_dump_path:
            return bool(set(expect_attr_dict.items()).issubset(node_attr.items()))
    return False

def check_tensordump_num_from_ir(graph_dir):
    cnt = 0
    if not os.path.exists(graph_dir):
        logger.critical(f"When executing ut of tensordump, directory: {graph_dir} is not exist")
    validate_ir = glob.glob(f"{graph_dir}/rank_0/*_validate_*.ir")[0]
    with open(validate_ir, "r", encoding="utf-8") as file:
        for line in file:
            cnt += 1 if "TensorDump(" in line else 0
    return cnt


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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    assert tensordump_num == 2

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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    assert tensordump_num == 4


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
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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


def test_cell_level_dump_in_with_certain_cell_no_side_effect_tensordump():
    """
    Feature: test no_side_effect tensordump only in SoftmaxCell.
    test tensordump op behavior in scenario of dump single cell input
    Description: Test dump SoftmaxCell input
    Expectation: compile success
    """
    class SoftmaxCellWrapper(nn.Cell):
        def __init__(self):
            super(SoftmaxCellWrapper, self).__init__()
            self.softmax = SoftmaxCell()
            self.no_side_effect_td = ops.TensorDump('in')
            self.no_side_effect_td.add_prim_attr("side_effect_io", False)
        def construct(self, x):
            depended = ops.depend(x, self.no_side_effect_td("softmax_input_dump.npy", x))
            out = self.softmax(depended)
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
            self.dump = ops.TensorDump('in')
            self.dump.add_prim_attr("def_attr", True)

        def construct(self, x):
            x = self.matmul1(x)
            sft = self.softmax(x)
            out1 = self.matmul2(x)
            out1 = self.add(sft, out1)
            self.dump("dump_out1.npy", out1)
            out1 = ops.relu(out1)
            result = self.matmul3(out1)
            return result
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
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
    tensordump_node_infos = get_tensordump_node_infos(validator)
    assert tensordump_num == 2
    assert check_dump_path_and_attr(tensordump_node_infos, "softmax_input_dump_in.npy", {})
    assert check_dump_path_and_attr(tensordump_node_infos, "dump_out1_in.npy", {"def_attr": True})


def test_forward_communication_no_side_effect_tensordump():
    """
    Feature: test tensordump with forward communication
    Description: test tensordump with mode 'out' for operators
    which need forward communication process in distributed scene.
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.add = P.Add()
            self.no_side_effect_td = ops.TensorDump('out')
            self.no_side_effect_td.add_prim_attr("side_effect_io", False)

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            out1 = ops.depend(out1, self.no_side_effect_td("out1_before_allreduce.npy", out1))
            out2 = self.matmul2(out1, b)
            out3 = self.matmul3(out1, b)
            out4 = self.add(out2, out3)
            return out4

    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((1, 8), (8, 1))
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
    assert tensordump_num == 1


def test_forward_communication_fwddump_and_bwddump():
    """
    Feature: test tensordump with forward communication
    Description: test tensordump with mode 'out' for operators
    which need forward communication process in distributed scene.
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.add = P.Add()
            self.no_side_effect_td = ops.TensorDump('out')
            self.no_side_effect_td.add_prim_attr("side_effect_io", False)
            self.bwd_tensordump_hook_in = ops.DumpGradient()

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            out1 = self.bwd_tensordump_hook_in("hook_before_out1.npy", out1, 'in')
            out1 = ops.depend(out1, self.no_side_effect_td("out1_before_allreduce.npy", out1))
            out2 = self.matmul2(out1, b)
            out3 = self.matmul3(out1, b)
            out4 = self.add(out2, out3)
            return out4
    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((1, 8), (8, 1))
    strategy2 = ((2, 4), (4, 1))
    strategy3 = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3=strategy3)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    _ = compile_net(net, x, y, b)
    tensordump_num = check_tensordump_num_from_ir(graph_path)
    assert tensordump_num == 2


def test_forward_communication_continuous_fwddump_and_bwddump():
    """
    Feature: test tensordump with forward communication
    Description: test tensordump with mode 'out' for operators
    which need forward communication process in distributed scene.
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.add = P.Add()
            self.no_side_effect_td = ops.TensorDump('out')
            self.no_side_effect_td.add_prim_attr("side_effect_io", False)
            self.bwd_tensordump_hook_in = ops.DumpGradient()

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            out1 = self.bwd_tensordump_hook_in("hook1_before_out1.npy", out1, 'in')
            out1 = ops.depend(out1, self.no_side_effect_td("out1_before_allreduce1.npy", out1))
            out1 = self.bwd_tensordump_hook_in("hook1_before_out2.npy", out1, 'in')
            out1 = ops.depend(out1, self.no_side_effect_td("out1_before_allreduce2.npy", out1))
            out2 = self.matmul2(out1, b)
            out3 = self.matmul3(out1, b)
            out4 = self.add(out2, out3)
            return out4

    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((1, 8), (8, 1))
    strategy2 = ((2, 4), (4, 1))
    strategy3 = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3=strategy3)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    _ = compile_net(net, x, y, b)
    tensordump_num = check_tensordump_num_from_ir(graph_path)
    assert tensordump_num == 4

def test_forward_communication_multioutput_fwddump_and_bwddump():
    """
    Feature: test tensordump with forward communication
    Description: test tensordump with mode 'out' for operators
    which need forward communication process in distributed scene.
    Expectation: compile success
    """
    class GroupedMatmulNetSplit0Weight(nn.Cell):
        def __init__(self, np_w0, np_w1, split_item=3, group_type=0, mul_stra=None, gmm_stra=None, relu_stra=None):
            super().__init__()
            self.w = [Parameter(ms.Tensor(np_w0), "w0"), Parameter(ms.Tensor(np_w1), "w1")]
            self.b = None
            self.scale = None
            self.offset = None
            self.antiquant_scale = None
            self.antiquant_offset = None

            self.mul = ops.Mul().shard(mul_stra)
            self.gmm = GroupedMatmul(split_item, group_type).shard(gmm_stra)
            self.relu1 = ops.ReLU().shard(relu_stra)
            self.relu2 = ops.ReLU().shard(relu_stra)
            self.no_side_effect_td = ops.TensorDump('out')
            self.no_side_effect_td.add_prim_attr("side_effect_io", False)
            self.bwd_tensordump_hook_in = ops.DumpGradient()

        def construct(self, x0, x1, group_list, one0, one1):
            x0 = self.mul(x0, one0)
            x1 = self.mul(x1, one1)
            x = [x0, x1]

            out = self.gmm(x, self.w, self.b, self.scale, self.offset,\
                        self.antiquant_scale, self.antiquant_offset, group_list)
            out0 = out[0]
            out1 = out[1]
            out0 = self.bwd_tensordump_hook_in("out0_hook0", out0, 'in')
            out0 = ops.depend(out0, self.no_side_effect_td("out0_dump0", out0))
            out0 = self.bwd_tensordump_hook_in("out0_hook1", out0, 'in')
            out0 = ops.depend(out0, self.no_side_effect_td("out0_dump1", out0))
            out0 = self.relu1(out0)

            out1 = self.bwd_tensordump_hook_in("out1_hook0", out1, 'in')
            out1 = ops.depend(out1, self.no_side_effect_td("out1_dump0", out1))
            out1 = self.bwd_tensordump_hook_in("out1_hook1", out1, 'in')
            out1 = ops.depend(out1, self.no_side_effect_td("out1_dump1", out1))
            out1 = self.relu2(out1)
            out = [out0, out1]
            return out

    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=4, global_rank=0)
    context.set_auto_parallel_context(dataset_strategy="full_batch")
    context.set_context(device_target="Ascend", mode=ms.GRAPH_MODE)
    mp = 4
    mul_stra = ((1, 1), (1, 1))
    gmm_stra = (((1, mp),) * 2, ((mp, 1),) * 2, ((),), ((),), ((),), ((),), ((),), ())  # x,w / b 4 quant + grouplist
    relu_stra = ((1, mp),)

    M0 = 16
    K0 = 256
    N0 = 128
    np_x0 = np.random.uniform(0.1, 2, size=[M0, K0]).astype(np.float16)
    np_w0 = np.random.uniform(0.1, 1, size=[K0, N0]).astype(np.float16)

    gmm_net = GroupedMatmulNetSplit0Weight(np_w0, np_w0, split_item=0, group_type=-1,
                                           mul_stra=mul_stra, gmm_stra=gmm_stra, relu_stra=relu_stra)
    # ms calculate
    x0 = ms.Tensor(np_x0)
    x1 = ms.Tensor(np_x0)
    group_list = None
    one0 = ms.Tensor(np.ones_like(np_x0).astype(np.float16))
    one1 = ms.Tensor(np.ones_like(np_x0).astype(np.float16))

    gmm_net.set_inputs(x0, x1, group_list, one0, one1)
    phase = compile_net(gmm_net, x0, x1, group_list, one0, one1)
    validator = ParallelValidator(gmm_net, phase)
    tensordump_num = check_tensordump_num_from_ir(graph_path)
    assert validator.check_parameter_shape('w0', [K0/mp, N0])
    assert validator.check_parameter_shape('w1', [K0/mp, N0])
    assert tensordump_num == 4

def test_redistribution_fwddump_and_bwddump():
    """
    Feature: test tensordump with parallel redistribution
    Description: test tensordump with mode 'in' and bwd hook with mode 'out'
    which need redistribution process in distributed scene.
    Expectation: compile success
    """
    class Net(nn.Cell):
        def __init__(self, strategy1, strategy2, strategy3):
            super().__init__()
            self.matmul1 = P.MatMul().shard(strategy1)
            self.matmul2 = P.MatMul().shard(strategy2)
            self.matmul3 = P.MatMul().shard(strategy3)
            self.add = P.Add()
            self.no_side_effect_td_in = ops.TensorDump('in')
            self.no_side_effect_td_in.add_prim_attr("side_effect_io", False)
            self.bwd_tensordump_hook_out = ops.DumpGradient()

        def construct(self, x, y, b):
            out1 = self.matmul1(x, y)
            out1 = self.bwd_tensordump_hook_out("hook1_after_redist_1.npy", out1, 'out')
            out1 = ops.depend(out1, self.no_side_effect_td_in("out1_after_redist_1.npy", out1))
            out1 = self.bwd_tensordump_hook_out("hook1_after_redist_1.npy", out1, 'out')
            out1 = ops.depend(out1, self.no_side_effect_td_in("out1_after_redist_2.npy", out1))
            out2 = self.matmul2(out1, b)
            out3 = self.matmul3(out1, b)
            out4 = self.add(out2, out3)
            return out4

    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    strategy1 = ((1, 8), (8, 1))
    strategy2 = ((2, 4), (4, 1))
    strategy3 = ((2, 2), (2, 2))
    net = GradWrap(NetWithLoss(Net(strategy1, strategy2, strategy3=strategy3)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')

    x = Tensor(np.ones([128, 32]), dtype=ms.float32)
    y = Tensor(np.ones([32, 64]), dtype=ms.float32)
    b = Tensor(np.ones([64, 64]), dtype=ms.float32)

    _ = compile_net(net, x, y, b)
    tensordump_num = check_tensordump_num_from_ir(graph_path)
    assert tensordump_num == 8


def test_splittensor_fwddump_and_bwddump():
    """
    Feature: test dump with SplitTensor
    Description: test dump with split tensor and whether 'concatnet_out.npy' will be suffixed with 'in'
    Expectation: compile success
    """
    class ConcatNet(nn.Cell):
        def __init__(self, concat_strategy, axis=0):
            super().__init__()
            self.concat = P.Concat(axis=axis).shard(concat_strategy)
            self.zero_pad_size = (2, 4096, 8, 64)
            self.zero_pad = Tensor(np.zeros(self.zero_pad_size), dtype=ms.float16)
            self.dg = ops.DumpGradient()
            self.no_side_effect_td_in = ops.TensorDump('in')
            self.no_side_effect_td_in.add_prim_attr("side_effect_io", False)

        def construct(self, x):
            x = self.dg("grad_x.npy", x, 'in')
            out = ops.depend(x, self.no_side_effect_td_in("concatnet_out.npy", x))
            return out

    graph_path = gen_save_graph_path_by_testcase(sys._getframe(0).f_code.co_name)
    context.set_context(save_graphs=2, save_graphs_path=graph_path)
    context.set_auto_parallel_context(device_num=8, global_rank=0, gradients_mean=True)
    cc_stra = ((2, 1, 2, 1), (2, 1, 2, 1))
    net = GradWrap(NetWithLoss(ConcatNet(cc_stra, axis=3)))
    context.set_auto_parallel_context(parallel_mode='semi_auto_parallel')
    x = Tensor(np.zeros((2, 4096, 8, 128)), dtype=ms.float16)
    phase = compile_net(net, x)
    validator = ParallelValidator(net, phase)
    tensordump_num = get_tensordump_node_num(validator)
    tensordump_node_infos = get_tensordump_node_infos(validator)
    dump_gradient_node_infos = get_tensordump_node_infos(validator, 'DumpGradient')
    tensordump_num = check_tensordump_num_from_ir(graph_path)
    assert tensordump_num == 2
    assert check_dump_path_and_attr(tensordump_node_infos, "concatnet_out_in.npy", {})
    assert check_dump_path_and_attr(dump_gradient_node_infos, "grad_x_in.npy", {})
