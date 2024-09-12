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
# ============================================================================
"""Test graph_utils.cc model"""
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Symbol
from mindspore.ops import operations as P
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore import context, Tensor, Parameter
from mindspore.parallel.shard import Layout
from mindspore.context import ParallelMode
from parallel.utils.utils import compile_net, ParallelValidator


def setup_function():
    context.reset_auto_parallel_context()
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


def test_parallel_dynamic_shape_with_features_007():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Corresponding test case is test_parallel_dynamic_shape_with_features_007.
    Expectation: Compile success and assertion passed.
    """

    class MatMulNet(nn.Cell):
        def __init__(self):
            super().__init__()
            self.matmul1 = P.MatMul()
            self.matmul2 = P.MatMul()
            self.matmul1_weight = Parameter(np.full((64, 32), 0.5, dtype=np.float32), name="weight1")
            self.matmul2_weight = Parameter(np.full((32, 32), 0.8, dtype=np.float32), name="weight2")
            self.matmul1.shard(((4, 1), (1, 2)))
            self.matmul2.shard(((2, 2), (2, 2)))
            self.relu = nn.ReLU()

        def construct(self, inputs):
            x = self.matmul1(inputs, self.matmul1_weight)
            x = self.matmul2(x, self.matmul2_weight)
            x = self.relu(x)
            return x

    dump_ir_path = "./test_parallel_dynamic_shape_with_features_007"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, full_batch=True,
                                      dataset_strategy=(dataset_shard,))
    model = MatMulNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s0 = Symbol(divisor=8)
    x = Tensor(shape=[s0, 64], dtype=mstype.float32)
    model.set_inputs(x)
    phase = compile_net(model, x)
    _ = ParallelValidator(model, phase)


class ReShapeNet(nn.Cell):
    def __init__(self):
        super(ReShapeNet, self).__init__()
        self.weight = Parameter(np.full((1, 1, 1, 1), 0.5, dtype=np.float16), name="weight")
        self.add = P.Add()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.relu = P.ReLU()
        self.add.shard(((2, 2, 2, 1), (1, 1, 1, 1)))
        self.relu.shard(((4, 1),))

    def construct(self, x):
        x = self.add(x, self.weight)
        x_shape = self.shape(x)
        x = self.reshape(x, (x_shape[0] * x_shape[1], x_shape[2] * x_shape[3]))
        x = self.relu(x)
        return x


def test_parallel_dynamic_shape_with_features_010():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Corresponding test case is test_parallel_dynamic_shape_with_features_010.
    Expectation: Compile success and assertion passed.
    """

    dump_ir_path = "./test_parallel_dynamic_shape_with_features_010"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (8, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ReShapeNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s1 = Symbol(divisor=4)
    x = Tensor(shape=(None, 32, s1, 16), dtype=mstype.float16)
    model.set_inputs(x)
    phase = compile_net(model, x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs('Shape-0', ['TupleGetItem-0'])
    assert validator.check_node_inputs('redistribution_op_getitem-0', ['Shape-0', 0])
    assert validator.check_node_inputs('redistribution_op_getitem-1', ['Shape-0', 2])
    assert validator.check_node_inputs('MakeTuple-1',
                                       [1, 1, 'redistribution_op_getitem-0', 32, 'redistribution_op_getitem-1', 16])
    assert validator.check_node_inputs('Reshape-0', ['TupleGetItem-0', 'MakeTuple-1'])
    assert validator.check_node_inputs('AllGather-0', ['Reshape-0'])
    assert validator.check_node_inputs('ScalarMul-0', ['redistribution_op_getitem-0', 4])
    assert validator.check_node_inputs('ScalarFloorDiv-0', ['redistribution_op_getitem-1', 2])
    assert validator.check_node_inputs('MakeTuple-4', ['ScalarMul-0', 16, 'ScalarFloorDiv-0', 16])
    assert validator.check_node_inputs('Reshape-1', ['TupleGetItem-6', 'MakeTuple-4'])
    assert validator.check_node_inputs('Add-0', ['Reshape-1', 'Load-0'])
    assert validator.check_node_inputs('AllGather-2', ['Add-0'])
    assert validator.check_node_inputs('AllGather-3', ['AllGather-2'])
    assert validator.check_node_inputs('Split-4', ['AllGather-3', 0, 2])
    assert validator.check_node_inputs('TupleGetItem-7', ['Split-4', 0])
    assert validator.check_node_inputs('TupleGetItem-8', ['Split-4', 1])
    assert validator.check_node_inputs('MakeTuple-5', ['TupleGetItem-7', 'TupleGetItem-8'])
    assert validator.check_node_inputs('Concat-2', ['MakeTuple-5', 1])
    assert validator.check_node_inputs('AllGather-4', ['Concat-2'])
    assert validator.check_node_inputs('Split-5', ['AllGather-4', 0, 2])
    assert validator.check_node_inputs('TupleGetItem-9', ['Split-5', 0])
    assert validator.check_node_inputs('TupleGetItem-10', ['Split-5', 1])
    assert validator.check_node_inputs('MakeTuple-6', ['TupleGetItem-9', 'TupleGetItem-10'])
    assert validator.check_node_inputs('Concat-3', ['MakeTuple-6', 2])
    assert validator.check_node_inputs('Shape-1', ['Add-0'])
    assert validator.check_node_inputs('TupleGetItem-11', ['Shape-1', 0])
    assert validator.check_node_inputs('ScalarMul-1', ['TupleGetItem-11', 2])
    assert validator.check_node_inputs('ScalarMul-2', ['ScalarMul-1', 32])
    assert validator.check_node_inputs('TupleGetItem-12', ['Shape-1', 2])
    assert validator.check_node_inputs('ScalarMul-3', ['TupleGetItem-12', 2])
    assert validator.check_node_inputs('ScalarMul-4', ['ScalarMul-3', 16])
    assert validator.check_node_inputs('MakeTuple-7', ['ScalarMul-2', 'ScalarMul-4'])
    assert validator.check_node_inputs('Reshape-2', ['Concat-3', 'MakeTuple-7'])
    assert validator.check_node_inputs('Split-6', ['Reshape-2', 0, 4])
    assert validator.check_node_inputs('TupleGetItem-13', ['Split-6', 0])
    assert validator.check_node_inputs('ReLU-0', ['TupleGetItem-13'])


def test_parallel_reshape_has_multi_dynamic_axis():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Reshape has more than one dynamic axis.
    Expectation: Compile success and assertion passed.
    """

    dump_ir_path = "./test_parallel_reshape_has_multi_dynamic_axis"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (1, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, full_batch=True,
                                      dataset_strategy=(dataset_shard,))
    model = ReShapeNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    # 4,32,16,16
    s0 = Symbol(divisor=8)
    s1 = Symbol(divisor=4)
    x = Tensor(shape=[s0, 32, s1, 16], dtype=mstype.float16)
    model.set_inputs(x)
    phase = compile_net(model, x)
    _ = ParallelValidator(model, phase)


def test_parallel_static_reshape_has_multi_user():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Reshape has more than one dynamic axis.
    Expectation: Compile success and assertion passed.
    """

    class NewReshapeNet(ReShapeNet):
        def __init__(self):
            super(NewReshapeNet, self).__init__()
            self.relu0 = P.ReLU().shard(((2, 4),))
            self.relu1 = P.ReLU().shard(((4, 2),))

        def construct(self, inputs):
            x_shape = self.shape(inputs)
            x = self.reshape(inputs, (x_shape[0] * x_shape[1], x_shape[2] * x_shape[3]))
            y1 = self.relu0(x)  # shard: (2, 4)
            y2 = self.relu1(x)  # shard: (4, 2)
            return y1 + y2

    dump_ir_path = "./test_parallel_static_reshape_has_multi_user"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (1, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8, full_batch=True,
                                      dataset_strategy=(dataset_shard,))
    model = NewReshapeNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    # 4,32,16,16
    x = Tensor(np.random.rand(3, 32, 5, 16), dtype=mstype.float16)
    model.set_inputs(x)
    phase = compile_net(model, x)
    _ = ParallelValidator(model, phase)


def test_parallel_dynamic_shape_with_features_011():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Corresponding test case is test_parallel_dynamic_shape_with_features_011.
    Expectation: Compile success and assertion passed.
    """

    dump_ir_path = "./test_parallel_dynamic_shape_with_features_011"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (8, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = ReShapeNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s1 = Symbol(divisor=4)
    input_dx = Tensor(shape=[s1, s1, s1, s1], dtype=mstype.float16)
    model.set_inputs(input_dx)
    phase = compile_net(model, input_dx)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs('Shape-0', ['TupleGetItem-0'])
    assert validator.check_node_inputs('redistribution_op_getitem-0', ['Shape-0', 0])
    assert validator.check_node_inputs('redistribution_op_getitem-1', ['Shape-0', 1])
    assert validator.check_node_inputs('redistribution_op_getitem-2', ['Shape-0', 2])
    assert validator.check_node_inputs('redistribution_op_getitem-3', ['Shape-0', 3])
    assert validator.check_node_inputs('MakeTuple-1',
                                       [1, 1, 'redistribution_op_getitem-0', 'redistribution_op_getitem-1',
                                        'redistribution_op_getitem-2', 'redistribution_op_getitem-3'])
    assert validator.check_node_inputs('Reshape-0', ['TupleGetItem-0', 'MakeTuple-1'])
    assert validator.check_node_inputs('ScalarMul-0', ['redistribution_op_getitem-0', 4])
    assert validator.check_node_inputs('ScalarFloorDiv-0', ['redistribution_op_getitem-1', 2])
    assert validator.check_node_inputs('ScalarFloorDiv-1', ['redistribution_op_getitem-2', 2])
    assert validator.check_node_inputs('MakeTuple-4', ['ScalarMul-0', 'ScalarFloorDiv-0', 'ScalarFloorDiv-1',
                                                       'redistribution_op_getitem-3'])
    assert validator.check_node_inputs('Reshape-1', ['TupleGetItem-6', 'MakeTuple-4'])
    assert validator.check_node_inputs('Shape-1', ['Add-0'])
    assert validator.check_node_inputs('TupleGetItem-11', ['Shape-1', 0])
    assert validator.check_node_inputs('ScalarMul-1', ['TupleGetItem-11', 2])
    assert validator.check_node_inputs('TupleGetItem-12', ['Shape-1', 1])
    assert validator.check_node_inputs('ScalarMul-2', ['TupleGetItem-12', 2])
    assert validator.check_node_inputs('ScalarMul-3', ['ScalarMul-1', 'ScalarMul-2'])
    assert validator.check_node_inputs('TupleGetItem-13', ['Shape-1', 2])
    assert validator.check_node_inputs('ScalarMul-4', ['TupleGetItem-13', 2])
    assert validator.check_node_inputs('TupleGetItem-14', ['Shape-1', 3])
    assert validator.check_node_inputs('ScalarMul-5', ['ScalarMul-4', 'TupleGetItem-14'])
    assert validator.check_node_inputs('MakeTuple-7', ['ScalarMul-3', 'ScalarMul-5'])
    assert validator.check_node_inputs('Reshape-2', ['Concat-3', 'MakeTuple-7'])


def test_parallel_dynamic_shape_with_features_013():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Corresponding test case is test_parallel_dynamic_shape_with_features_013.
    Expectation: Compile success and assertion passed.
    """

    class NewNet(ReShapeNet):
        def __init__(self):
            super(NewNet, self).__init__()

        def construct(self, inputs):
            x = self.add(inputs, self.weight)
            x_shape = self.shape(x)
            x = self.reshape(x, (-1, x_shape[2] * x_shape[3]))
            x = self.relu(x)
            return x

    dump_ir_path = "./test_parallel_dynamic_shape_with_features_013"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (8, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = NewNet()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s1 = Symbol(divisor=8)
    input_dx = Tensor(shape=[None, 32, s1, 16], dtype=mstype.float16)
    model.set_inputs(input_dx)
    phase = compile_net(model, input_dx)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs('ScalarMul-0', ['redistribution_op_getitem-0', 4])
    assert validator.check_node_inputs('ScalarFloorDiv-0', ['redistribution_op_getitem-1', 2])
    assert validator.check_node_inputs('MakeTuple-4', ['ScalarMul-0', 16, 'ScalarFloorDiv-0', 16])
    assert validator.check_node_inputs('Reshape-1', ['TupleGetItem-6', 'MakeTuple-4'])
    assert validator.check_node_inputs('Shape-1', ['Add-0'])
    assert validator.check_node_inputs('TupleGetItem-11', ['Shape-1', 2])
    assert validator.check_node_inputs('ScalarMul-1', ['TupleGetItem-11', 2])
    assert validator.check_node_inputs('ScalarMul-2', ['ScalarMul-1', 16])
    assert validator.check_node_inputs('MakeTuple-7', [-1, 'ScalarMul-2'])
    assert validator.check_node_inputs('Reshape-2', ['Concat-3', 'MakeTuple-7'])


def test_parallel_dynamic_shape_with_features_015():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Corresponding test case is test_parallel_dynamic_shape_with_features_015.
    Expectation: Compile success and assertion passed.
    """

    class TwoReShapeNet1(ReShapeNet):
        def __init__(self):
            super(TwoReShapeNet1, self).__init__()
            self.reshape2 = P.Reshape()
            self.mean = P.ReduceMean(keep_dims=False)

        def construct(self, inputs):
            x = self.add(inputs, self.weight)
            x_shape = self.shape(x)
            x = self.reshape(x, (x_shape[0] * x_shape[1], x_shape[2] * x_shape[3]))
            x = self.relu(x)
            x = self.reshape2(x, (-1, 2, 8, 4))
            x = self.mean(x, (2, 3))
            x = self.relu(x)
            return x

    dump_ir_path = "./test_parallel_dynamic_shape_with_features_015"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (8, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = TwoReShapeNet1()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s1 = Symbol(divisor=4)
    input_dx = Tensor(shape=[None, s1, 8, 4], dtype=mstype.float32)
    model.set_inputs(input_dx)
    phase = compile_net(model, input_dx)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs('Shape-0', ['TupleGetItem-0'])
    assert validator.check_node_inputs('redistribution_op_getitem-0', ['Shape-0', 0])
    assert validator.check_node_inputs('redistribution_op_getitem-1', ['Shape-0', 1])
    assert validator.check_node_inputs('MakeTuple-1',
                                       [1, 1, 'redistribution_op_getitem-0', 'redistribution_op_getitem-1', 8, 4])
    assert validator.check_node_inputs('Reshape-0', ['TupleGetItem-0', 'MakeTuple-1'])
    assert validator.check_node_inputs('ScalarMul-0', ['redistribution_op_getitem-0', 4])
    assert validator.check_node_inputs('ScalarFloorDiv-0', ['redistribution_op_getitem-1', 2])
    assert validator.check_node_inputs('MakeTuple-4', ['ScalarMul-0', 'ScalarFloorDiv-0', 4, 4])
    assert validator.check_node_inputs('Reshape-1', ['TupleGetItem-6', 'MakeTuple-4'])
    assert validator.check_node_inputs('Concat-3', ['MakeTuple-6', 2])
    assert validator.check_node_inputs('Reshape-2', ['Concat-3', (-1, 32)])
    assert validator.check_node_inputs('Reshape-3', ['ReLU-0', (1, -1, 2, 8, 4)])
    assert validator.check_node_inputs('Reshape-4', ['TupleGetItem-12', (-1, 2, 8, 4)])
    assert validator.check_node_inputs('ReduceMean-0', ['Reshape-4', (2, 3), False])
    assert validator.check_node_inputs('Reshape-5', ['ReduceMean-0', (1, -1, 2)])
    assert validator.check_node_inputs('AllGather-5', ['Reshape-5'])
    assert validator.check_node_inputs('Split-8', ['AllGather-5', 0, 2])
    assert validator.check_node_inputs('TupleGetItem-13', ['Split-8', 0])
    assert validator.check_node_inputs('TupleGetItem-14', ['Split-8', 1])
    assert validator.check_node_inputs('MakeTuple-7', ['TupleGetItem-13', 'TupleGetItem-14'])
    assert validator.check_node_inputs('Concat-4', ['MakeTuple-7', 1])
    assert validator.check_node_inputs('Reshape-6', ['Concat-4', (-1, 2)])


def test_parallel_dynamic_shape_with_features_019():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Corresponding test case is test_parallel_dynamic_shape_with_features_019.
    Expectation: Compile success and assertion passed.
    """

    class TwoReShapeNet1(ReShapeNet):
        def __init__(self):
            super(TwoReShapeNet1, self).__init__()
            self.reshape2 = P.Reshape()
            self.mean = P.ReduceMean(keep_dims=False)

        def construct(self, inputs):
            x = self.add(inputs, self.weight)
            x_shape = self.shape(x)
            x = self.reshape(x, (x_shape[0] * x_shape[1], x_shape[2] * x_shape[3]))
            x = self.relu(x)
            x = self.reshape2(x, (-1, 2, 8, 4))
            x = self.mean(x, (2, 3))
            x = self.relu(x)
            return x

    dump_ir_path = "./test_parallel_dynamic_shape_with_features_019"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (8, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = TwoReShapeNet1()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s1 = Symbol(divisor=4)
    input_dx = Tensor(shape=[None, s1, 8, 4], dtype=mstype.float32)
    model.set_inputs(input_dx)
    phase = compile_net(model, input_dx)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs('ScalarMul-0', ['redistribution_op_getitem-0', 4])
    assert validator.check_node_inputs('ScalarFloorDiv-0', ['redistribution_op_getitem-1', 2])
    assert validator.check_node_inputs('MakeTuple-4', ['ScalarMul-0', 'ScalarFloorDiv-0', 4, 4])
    assert validator.check_node_inputs('Reshape-1', ['TupleGetItem-6', 'MakeTuple-4'])
    assert validator.check_node_inputs('Reshape-2', ['Concat-3', (-1, 32)])
    assert validator.check_node_inputs('Reshape-3', ['ReLU-0', (1, -1, 2, 8, 4)])
    assert validator.check_node_inputs('Reshape-4', ['TupleGetItem-12', (-1, 2, 8, 4)])
    assert validator.check_node_inputs('Reshape-5', ['ReduceMean-0', (1, -1, 2)])
    assert validator.check_node_inputs('Reshape-6', ['Concat-4', (-1, 2)])


def test_parallel_dynamic_shape_with_features_022():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Corresponding test case is test_parallel_dynamic_shape_with_features_022.
    Expectation: Compile success and assertion passed.
    """

    class TwoReShapeNet1(ReShapeNet):
        def __init__(self):
            super(TwoReShapeNet1, self).__init__()
            self.reshape2 = P.Reshape()
            self.mean = P.ReduceMean(keep_dims=False)
            self.relu.shard(((2, 4),))

        def construct(self, inputs):
            x = self.add(inputs, self.weight)
            x_shape = self.shape(x)
            x = self.reshape(x, (x_shape[0] * x_shape[1], x_shape[2] * x_shape[3]))
            x = self.relu(x)
            x = self.reshape2(x, (2, -1, 2, 2))
            x = self.mean(x, (2, 3))
            x = self.relu(x)
            return x

    dump_ir_path = "./test_parallel_dynamic_shape_with_features_022"
    context.set_context(save_graphs=True, save_graphs_path=dump_ir_path)
    dataset_shard = (8, 1, 1, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    model = TwoReShapeNet1()
    model = _VirtualDatasetCell(model)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s1 = Symbol(divisor=16)
    s2 = Symbol(divisor=8)
    input_dx = Tensor(shape=[None, s1, s2, s1], dtype=mstype.float32)
    model.set_inputs(input_dx)
    phase = compile_net(model, input_dx)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs('redistribution_op_getitem-0', ['Shape-0', 0])
    assert validator.check_node_inputs('redistribution_op_getitem-1', ['Shape-0', 1])
    assert validator.check_node_inputs('redistribution_op_getitem-2', ['Shape-0', 2])
    assert validator.check_node_inputs('redistribution_op_getitem-3', ['Shape-0', 3])
    assert validator.check_node_inputs('MakeTuple-1',
                                       [1, 1, 'redistribution_op_getitem-0', 'redistribution_op_getitem-1',
                                        'redistribution_op_getitem-2', 'redistribution_op_getitem-3'])
    assert validator.check_node_inputs('Reshape-0', ['TupleGetItem-0', 'MakeTuple-1'])
    assert validator.check_node_inputs('ScalarMul-0', ['redistribution_op_getitem-0', 4])
    assert validator.check_node_inputs('ScalarFloorDiv-0', ['redistribution_op_getitem-1', 2])
    assert validator.check_node_inputs('ScalarFloorDiv-1', ['redistribution_op_getitem-2', 2])
    assert validator.check_node_inputs('MakeTuple-4', ['ScalarMul-0', 'ScalarFloorDiv-0', 'ScalarFloorDiv-1',
                                                       'redistribution_op_getitem-3'])
    assert validator.check_node_inputs('Reshape-1', ['TupleGetItem-6', 'MakeTuple-4'])
    assert validator.check_node_inputs('Shape-1', ['Add-0'])
    assert validator.check_node_inputs('TupleGetItem-11', ['Shape-1', 0])
    assert validator.check_node_inputs('ScalarMul-1', ['TupleGetItem-11', 2])
    assert validator.check_node_inputs('TupleGetItem-12', ['Shape-1', 1])
    assert validator.check_node_inputs('ScalarMul-2', ['TupleGetItem-12', 2])
    assert validator.check_node_inputs('ScalarMul-3', ['ScalarMul-1', 'ScalarMul-2'])
    assert validator.check_node_inputs('TupleGetItem-13', ['Shape-1', 2])
    assert validator.check_node_inputs('ScalarMul-4', ['TupleGetItem-13', 2])
    assert validator.check_node_inputs('TupleGetItem-14', ['Shape-1', 3])
    assert validator.check_node_inputs('ScalarMul-5', ['ScalarMul-4', 'TupleGetItem-14'])
    assert validator.check_node_inputs('MakeTuple-7', ['ScalarMul-3', 'ScalarMul-5'])
    assert validator.check_node_inputs('Reshape-2', ['Concat-3', 'MakeTuple-7'])
    assert validator.check_node_inputs('Split-6', ['Reshape-2', 0, 2])
    assert validator.check_node_inputs('TupleGetItem-15', ['Split-6', 0])
    assert validator.check_node_inputs('Split-7', ['TupleGetItem-15', 1, 4])
    assert validator.check_node_inputs('TupleGetItem-16', ['Split-7', 0])
    assert validator.check_node_inputs('ReLU-0', ['TupleGetItem-16'])
    assert validator.check_node_inputs('AllGather-5', ['ReLU-0'])
    assert validator.check_node_inputs('AllGather-6', ['AllGather-5'])
    assert validator.check_node_inputs('Split-8', ['AllGather-6', 0, 4])
    assert validator.check_node_inputs('Reshape-3', ['Concat-4', (2, -1, 2, 2)])
    assert validator.check_node_inputs('Split-9', ['Reshape-3', 0, 2])
    assert validator.check_node_inputs('TupleGetItem-21', ['Split-9', 0])
    assert validator.check_node_inputs('ReduceMean-0', ['TupleGetItem-21', (2, 3), False])
    assert validator.check_node_inputs('Split-10', ['ReduceMean-0', 1, 4])
    assert validator.check_node_inputs('TupleGetItem-22', ['Split-10', 0])
    assert validator.check_node_inputs('ReLU-1', ['TupleGetItem-22'])


def test_reshape_on_shard_shape():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Test const dim in shape op shard.
    Expectation: Compile success and assertion passed.
    """

    class AttentionNet2(nn.Cell):
        def __init__(self, weight, bias, strategy1=None, strategy2=None, strategy3=None, strategy4=None):
            super(AttentionNet2, self).__init__()
            self.matmul = P.MatMul().shard(strategy1)
            self.weight = Parameter(weight, "w1")
            self.add = P.Add().shard(strategy2)
            self.bias = Parameter(bias, "bias")
            self.reshape = P.Reshape()  # .add_prim_attr("skip_redistribution", True)
            self.transpose = P.Transpose().shard(strategy3)
            self.realdiv = P.RealDiv().shard(strategy4)
            self.cast = P.Cast()

        def construct(self, x):
            out = self.add(x, self.bias)
            s = P.Shape()(out)
            out = self.reshape(out, (16 * 8, s[2], s[3]))
            out = self.transpose(out, (0, 2, 1))
            out = self.cast(out, mstype.float16)
            out = self.reshape(out, s)  # failed
            # out = self.reshape(out, (8, 16, s[2], s[3]))  # ok
            out = self.realdiv(out, 1.0)
            return out

    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8, global_rank=0)
    context.set_context(save_graphs=True)
    strategy1 = ((1, 1), (1, 8))
    strategy2 = ((8, 1, 1, 1), (1,))
    strategy3 = ((8, 1, 1),)
    strategy4 = ((8, 1, 1, 1), ())
    weight = Tensor(np.ones([32, 64]), dtype=mstype.float32)
    bias = Tensor(np.ones([1]), dtype=mstype.float32)
    net = AttentionNet2(weight, bias, strategy1, strategy2, strategy3, strategy4)
    input_x = Tensor(shape=[8, 16, None, None], dtype=mstype.float32)
    net.set_inputs(input_x)
    phase = compile_net(net, input_x)
    validator = ParallelValidator(net, phase)
    assert validator.check_node_inputs('Add-0', ['TupleGetItem-0', 'Load-0'])
    assert validator.check_node_inputs('Shape-0', ['Add-0'])
    assert validator.check_node_inputs('TupleGetItem-1', ['Shape-0', 2])
    assert validator.check_node_inputs('TupleGetItem-2', ['Shape-0', 3])
    assert validator.check_node_inputs('MakeTuple-1', [16, 'TupleGetItem-1', 'TupleGetItem-2'])
    assert validator.check_node_inputs('Reshape-0', ['Add-0', 'MakeTuple-1'])
    assert validator.check_node_inputs('Transpose-0', ['Reshape-0', (0, 2, 1)])
    assert validator.check_node_inputs('Cast-0', ['Transpose-0', 42])
    assert validator.check_node_inputs('tuple_getitem_replace_reshape-0', ['Shape-0', 2])
    assert validator.check_node_inputs('tuple_getitem_replace_reshape-1', ['Shape-0', 3])
    assert validator.check_node_inputs('Shape-1',
                                       [8, 16, 'tuple_getitem_replace_reshape-0', 'tuple_getitem_replace_reshape-1'])
    assert validator.check_node_inputs('Reshape-1', ['Cast-0', 'Shape-1'])


def test_parallel_dynamic_shape_with_prime_dim_in_input_in_llava():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Test input with prime 577 in LLaVA.
    Expectation: Compile success and assertion passed.
    """

    class PrimeNet(nn.Cell):
        def __init__(self):
            super(PrimeNet, self).__init__()
            self.reshape = P.Reshape()
            self.relu1 = P.ReLU()
            self.relu1.shard(((1, 8, 1),))
            self.relu2 = P.ReLU()
            self.relu2.shard(((1, 1),))

        def construct(self, x):
            x = self.relu1(x)
            x = self.reshape(x, (-1, 1024))
            x = self.relu2(x)
            return x

    context.set_context(save_graphs=True,
                        save_graphs_path="test_parallel_dynamic_shape_with_prime_dim_in_input_in_llava")
    dataset_shard = (1, 8, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    net = PrimeNet()
    model = _VirtualDatasetCell(net)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s = Symbol(divisor=8)
    input_x = Tensor(shape=[577, s, 64], dtype=mstype.float32)
    model.set_inputs(input_x)
    phase = compile_net(model, input_x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs('AllGather-0', ['ReLU-0'])
    assert validator.check_node_inputs('Split-0', ['AllGather-0', 0, 8])
    assert validator.check_node_inputs('MakeTuple-1',
                                       ['TupleGetItem-1', 'TupleGetItem-2', 'TupleGetItem-3', 'TupleGetItem-4',
                                        'TupleGetItem-5', 'TupleGetItem-6', 'TupleGetItem-7', 'TupleGetItem-8'])
    assert validator.check_node_inputs('Concat-0', ['MakeTuple-1', 1])
    assert validator.check_node_inputs('Reshape-0', ['Concat-0', (-1, 1024)])
    assert validator.check_node_inputs('ReLU-1', ['Reshape-0'])


def test_parallel_dynamic_shape_with_prime_dim_in_input_in_videochat():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Test input with prime 768 in VideoChat2.
    Expectation: Compile success and assertion passed.
    """

    class PrimeNet(nn.Cell):
        def __init__(self):
            super(PrimeNet, self).__init__()
            self.reshape = P.Reshape()
            self.relu1 = P.ReLU()
            self.relu1.shard(((8, 1),))
            self.relu2 = P.ReLU()
            self.relu2.shard(((1, 1, 1, 1),))

        def construct(self, x):
            x = self.relu1(x)
            x = self.reshape(x, (160, -1, 12, 64))
            x = self.relu2(x)
            return x

    context.set_context(save_graphs=True,
                        save_graphs_path="test_parallel_dynamic_shape_with_prime_dim_in_input_in_videochat")
    dataset_shard = (8, 1)
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8,
                                      dataset_strategy=(dataset_shard,))
    net = PrimeNet()
    model = _VirtualDatasetCell(net)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s = Symbol(divisor=8)
    input_x = Tensor(shape=[s, 768], dtype=mstype.float32)
    model.set_inputs(input_x)
    phase = compile_net(model, input_x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs('TupleGetItem-0', ['MakeTuple-0', 0])
    assert validator.check_node_inputs('ReLU-0', ['TupleGetItem-0'])
    assert validator.check_node_inputs('Reshape-0', ['ReLU-0', (20, -1, 12, 64)])
    assert validator.check_node_inputs('AllGather-0', ['Reshape-0'])
    assert validator.check_node_inputs('ReLU-1', ['AllGather-0'])


def test_parallel_dynamic_shape_with_rmsnorm_001():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Test with from [-1, 16, -1]/[2,4,1] -> [-1, 16, 1, -1]/[8,1,1,1].
    Expectation: Compile success and assertion passed.
    """

    class RmsNormNet(nn.Cell):
        def __init__(self):
            super(RmsNormNet, self).__init__()
            mul_np = np.full((128, 96), 0.5, dtype=np.float32)
            self.mul_weight = Parameter(Tensor(mul_np), name="mul_weight")
            rms_np = np.full((96,), 0.5, dtype=np.float32)
            self.rms_weight = Parameter(Tensor(rms_np), name="rms_weight")
            self.mul = P.Mul()
            self.rmsnorm = P.RmsNorm(epsilon=1e-6)
            self.rmsnorm.shard(((4, 1), (1,)))

        def construct(self, inputs):
            x, _ = self.rmsnorm(inputs, self.rms_weight)
            x = self.mul(x, self.mul_weight)
            return x

    context.set_context(save_graphs=True, save_graphs_path="test_parallel_dynamic_shape_with_rmsnorm_001")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8)
    net = RmsNormNet()
    model = _VirtualDatasetCell(net)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    input_x = Tensor(shape=[None, None], dtype=mstype.float32)
    model.set_inputs(input_x)
    phase = compile_net(model, input_x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs('TupleGetItem-3', ['RmsNorm-0', 0])
    assert validator.check_node_inputs(
        'Reshape-2', ['TupleGetItem-3', (1, -1, 96)])


def test_parallel_dynamic_shape_with_split():
    """
    Feature: Test tensor redistribution in dynamic shape.
    Description: Test with from [-1, 16, -1]/[2,4,1] -> [-1, 16, 1, -1]/[8,1,1,1].
    Expectation: Compile success and assertion passed.
    """

    class ParallelSplitNet(nn.Cell):
        def __init__(self):
            super(ParallelSplitNet, self).__init__()
            self.split = P.Split(0, 1)
            self.add_input = Parameter(Tensor(np.ones((128, 128, 64)).astype(np.float32)), name="add")
            self.add = P.Add()
            layout = Layout((2, 4), ("dp", "mp"))
            in_strategy = (layout("None", "dp", "mp"),)
            self.split.shard(in_strategy=in_strategy)

        def construct(self, x):
            x = self.split(x)
            x = self.add(x[0], self.add_input)
            return x

    context.set_context(save_graphs=True, save_graphs_path="test_parallel_dynamic_shape_with_split")
    context.set_auto_parallel_context(parallel_mode=ParallelMode.SEMI_AUTO_PARALLEL,
                                      global_rank=0, device_num=8)
    net = ParallelSplitNet()
    model = _VirtualDatasetCell(net)
    model._virtual_dataset.add_prim_attr("repeat_dim_direct", "right")
    s1 = Symbol(divisor=2)
    s2 = Symbol(divisor=4)
    input_x = Tensor(shape=[None, s1, s2], dtype=mstype.float32)
    model.set_inputs(input_x)
    phase = compile_net(model, input_x)
    validator = ParallelValidator(model, phase)
    assert validator.check_node_inputs('Split-3', ['Reshape-1', 0, 1])
    assert validator.check_node_inputs('TupleGetItem-7', ['Split-3', 0])
    assert validator.check_node_inputs('Shape-1', ['TupleGetItem-7'])
