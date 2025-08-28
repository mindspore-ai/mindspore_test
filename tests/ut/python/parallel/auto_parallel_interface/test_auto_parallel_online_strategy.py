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
# limitations under the License

import numpy as np

from mindspore import Tensor, Parameter, ParameterTuple, nn, ops
from mindspore.common import lazy_inline
from mindspore.communication.management import get_rank
from mindspore.nn import L1Loss, PipelineCell
from mindspore.nn.utils import no_init_parameters
from mindspore.parallel.shard import Layout
from mindspore.train import Model
from mindspore.parallel.strategy import get_strategy_metadata, get_current_strategy_metadata, \
    enable_save_strategy_online
from parallel.auto_parallel_interface._utils_dataset import FakeData
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode


class WithLossCell(nn.Cell):
    @lazy_inline
    def __init__(self, backbone, loss_fn):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self._loss_fn = loss_fn
        self._get_attr_from_cell(backbone)

    def construct(self, data, label):
        out = self._backbone(data)
        return self._loss_fn(out, label)


class MatMulNet(nn.Cell):
    def __init__(self, matmul_weight, **kwargs):
        super().__init__()
        self.matmul1 = ops.MatMul()
        self.matmul2 = ops.MatMul()
        self.matmul1_weight = Parameter(matmul_weight[0], name="weight1")
        self.matmul2_weight = Parameter(matmul_weight[1], name="weight2")
        if kwargs.get("strategy1") is not None:
            self.matmul1.shard(kwargs.get("strategy1"))
        if kwargs.get("strategy2") is not None:
            self.matmul2.shard(kwargs.get("strategy2"))

    def construct(self, inputs):
        x = self.matmul1(inputs, self.matmul1_weight)
        x = self.matmul2(x, self.matmul2_weight)
        return x


class StageNet(nn.Cell):
    def __init__(self, weight_list, micro_size, **kwargs):
        super().__init__()
        self.micro_size = micro_size
        self.block = nn.CellList()
        self.add = ops.Add()
        self.weight_list = weight_list
        self.add_list = []
        self.relu_block = nn.CellList()
        for i in range(self.micro_size):
            cell = MatMulNet(weight_list[i], **kwargs)
            cell.pipeline_stage = i
            cell.matmul1.set_comm_fusion = i
            cell.matmul2.set_comm_fusion = i
            relu = nn.ReLU()
            relu.pipeline_stage = i
            self.relu_block.append(relu)
            self.block.append(cell)
            self.add_list.append(
                Parameter(Tensor(np.full((1, 16), 0.1, dtype=np.float32)), name=f"weight{i}"))
        self.add_tuple = ParameterTuple(self.add_list)

    def construct(self, x):
        for i in range(self.micro_size):
            x = self.block[i](x)
            x = self.relu_block[i](x)
            x = self.add(x, self.add_tuple[i])
        return x


def test_parallel_optimizer_pp2dp4_zero1_strategy_full_op():
    """
    Feature: shard with stategy, full shard.
    Description: enable pipeline.
    Expectation: compile success, validation pass.
    """
    np.random.seed(100)
    weight1 = Tensor(0.1 * np.random.randn(96, 16).astype(np.float32))
    weight2 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight3 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight4 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight_list = [[weight1, weight2], [weight3, weight4]]

    # set auto_parallel
    enable_save_strategy_online()
    init_hccl(global_rank=0, device_num=16)
    optim_cfg = {"optimizer_level": "level1", "optimizer_weight_shard_size": -1, "parallel_optimizer_threshold": 0}
    parallel_config = {"parallel_mode": "semi_auto", "pipeline_config": {"stages": 2},
                       "parallel_optimizer_config": optim_cfg}

    # compile net
    with no_init_parameters():
        parallel_net = StageNet(weight_list=weight_list, micro_size=2, strategy1=((2, 2), (2, 1)),
                                strategy2=((2, 2), (2, 1)))
        opt = nn.Momentum(learning_rate=0.00001, momentum=0.09, params=parallel_net.get_parameters())
    loss_fn = L1Loss()
    loss_net_parallel = WithLossCell(parallel_net, loss_fn)
    op_parallel_net = PipelineCell(loss_net_parallel, 2)
    train_net = set_parallel_mode(op_parallel_net, parallel_config)
    model = Model(network=train_net, optimizer=opt)

    parallel_dataset = FakeData(size=128, batch_size=128, image_size=(96,), num_classes=16)
    model.train(epoch=2, train_dataset=parallel_dataset, callbacks=None, dataset_sink_mode=False)

    # local info
    local_info = get_current_strategy_metadata(network=parallel_net)
    param_list = local_info[0]["0.matmul1_weight"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == ((1, 2, 0), -1)

    # global_info, rank_8
    global_info = get_strategy_metadata(network=parallel_net)
    param_list = global_info[8]["1.matmul1_weight"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == ((1, 2, 0), -1)


def test_parallel_optimizer_pp2dp4_zero1_layout_op2():
    """
    Feature: specific axis is shard repeatedly with layout, not full shard, shard size is 2.
    Description: enable pipeline, pp stage is 2.
    Expectation: compile success, validation pass.
    """
    np.random.seed(100)
    weight1 = Tensor(0.1 * np.random.randn(96, 16).astype(np.float32))
    weight2 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight3 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight4 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight_list = [[weight1, weight2], [weight3, weight4]]

    # set auto_parallel
    enable_save_strategy_online()
    init_hccl(global_rank=0, device_num=16)
    rank_id = get_rank()
    train_strategy_filename = f"./strategy_layout_rank_{rank_id}.ckpt"
    optim_cfg = {"optimizer_level": "level1", "optimizer_weight_shard_size": 2, "parallel_optimizer_threshold": 0}
    parallel_config = {"parallel_mode": "semi_auto", "pipeline_config": {"stages": 2},
                       "save_strategy_file_path": train_strategy_filename,
                       "parallel_optimizer_config": optim_cfg}

    # compile net
    layout = Layout((2, 2, 2), ("dp", "mp", "sp"))
    strategy1 = (layout("dp", ("mp", "sp")), layout(("mp", "sp"), "None"))
    strategy2 = (layout("dp", ("mp", "sp")), layout(("mp", "sp"), "None"))

    with no_init_parameters():
        parallel_net = StageNet(weight_list=weight_list, micro_size=2, strategy1=strategy1,
                                strategy2=strategy2)
        opt = nn.Momentum(learning_rate=0.00001, momentum=0.09, params=parallel_net.get_parameters())
    loss_fn = L1Loss()
    loss_net_parallel = WithLossCell(parallel_net, loss_fn)
    op_parallel_net = PipelineCell(loss_net_parallel, 2)
    train_net = set_parallel_mode(op_parallel_net, parallel_config)
    model = Model(network=train_net, optimizer=opt)

    parallel_dataset = FakeData(size=128, batch_size=128, image_size=(96,), num_classes=16)
    model.train(epoch=2, train_dataset=parallel_dataset, callbacks=None, dataset_sink_mode=False)

    # local info
    local_info = get_current_strategy_metadata(network=parallel_net)
    param_list = local_info[0]["0.matmul1_weight"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == ((1, 0, 2), -1)

    # specific rank
    rank_info = get_strategy_metadata(network=parallel_net, rank_id=7)
    param_list = rank_info[7]["0.matmul1_weight"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == ((1, 0, 2), -1)

    # global rank
    global_info = get_strategy_metadata(network=parallel_net)
    param_list = global_info[15]["1.matmul1_weight"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == ((1, 0, 2), -1)


def test_parallel_optimizer_pp2dp4_zero3_layout_op2():
    """
    Feature: specific axis is shard repeatedly with layout, not full shard, shard size is 2.
    Description: pipeline stage 2.
    Expectation: compile success, validation pass.
    """
    np.random.seed(100)
    weight1 = Tensor(0.1 * np.random.randn(96, 16).astype(np.float32))
    weight2 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight3 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight4 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight_list = [[weight1, weight2], [weight3, weight4]]

    # set auto_parallel
    enable_save_strategy_online()
    init_hccl(global_rank=0, device_num=32)
    optim_cfg = {"optimizer_level": "level3", "optimizer_weight_shard_size": 2, "parallel_optimizer_threshold": 0}
    parallel_config = {"parallel_mode": "semi_auto", "pipeline_config": {"stages": 2},
                       "parallel_optimizer_config": optim_cfg}

    # compile net
    layout = Layout((1, 2, 4, 2), ("dp", "mp", "np", "sp"))
    strategy1 = (layout("dp", ("mp", "sp")), layout(("mp", "sp"), "None"))
    strategy2 = (layout("dp", ("mp", "sp")), layout(("mp", "sp"), "None"))

    with no_init_parameters():
        parallel_net = StageNet(weight_list=weight_list, micro_size=2, strategy1=strategy1,
                                strategy2=strategy2)
        opt = nn.Momentum(learning_rate=0.00001, momentum=0.09, params=parallel_net.get_parameters())
    loss_fn = L1Loss()
    loss_net_parallel = WithLossCell(parallel_net, loss_fn)
    op_parallel_net = PipelineCell(loss_net_parallel, 2)
    train_net = set_parallel_mode(op_parallel_net, parallel_config)
    model = Model(network=train_net, optimizer=opt)

    parallel_dataset = FakeData(size=128, batch_size=128, image_size=(96,), num_classes=16)
    model.train(epoch=2, train_dataset=parallel_dataset, callbacks=None, dataset_sink_mode=False)

    # local info
    local_info = get_current_strategy_metadata(network=parallel_net)
    param_list = local_info[0]["0.matmul1_weight"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == ((3, 0, 1), -1)

    # specific rank
    rank_info = get_strategy_metadata(network=parallel_net, rank_id=7)
    param_list = rank_info[7]["weight0"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == (-1, -1)

    # global rank
    global_info = get_strategy_metadata(network=parallel_net)
    param_list = global_info[16]["1.matmul2_weight"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == ((3, 0, 1), -1)


def test_parallel_optimizer_pp1dp4_zero2_layout_full_op():
    """
    Feature: specific axis is shard repeatedly with layout, opt full shard.
    Description: do not enable pipeline, all remain axis are used for opt shard.
    Expectation: compile success, validation pass
    """
    np.random.seed(100)
    weight1 = Tensor(0.1 * np.random.randn(96, 16).astype(np.float32))
    weight2 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight3 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight4 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight_list = [[weight1, weight2], [weight3, weight4]]

    # set auto_parallel
    init_hccl(global_rank=0, device_num=16)
    enable_save_strategy_online()
    optim_cfg = {"optimizer_level": "level2", "optimizer_weight_shard_size": -1, "parallel_optimizer_threshold": 0}
    parallel_config = {"parallel_mode": "semi_auto", "pipeline_config": {"stages": 1},
                       "parallel_optimizer_config": optim_cfg}

    # compile net
    layout = Layout((1, 2, 4, 2), ("dp", "mp", "np", "sp"))
    strategy1 = (layout("dp", ("mp", "sp")), layout(("mp", "sp"), "None"))
    strategy2 = (layout("dp", ("mp", "sp")), layout(("mp", "sp"), "None"))

    with no_init_parameters():
        parallel_net = StageNet(weight_list=weight_list, micro_size=2, strategy1=strategy1,
                                strategy2=strategy2)
        opt = nn.Momentum(learning_rate=0.00001, momentum=0.09, params=parallel_net.get_parameters())
    loss_fn = L1Loss()
    loss_net_parallel = WithLossCell(parallel_net, loss_fn)
    op_parallel_net = PipelineCell(loss_net_parallel, 2)
    train_net = set_parallel_mode(op_parallel_net, parallel_config)
    model = Model(network=train_net, optimizer=opt)

    parallel_dataset = FakeData(size=128, batch_size=128, image_size=(96,), num_classes=16)
    model.train(epoch=2, train_dataset=parallel_dataset, callbacks=None, dataset_sink_mode=False)

    # local info
    local_info = get_current_strategy_metadata(network=train_net)
    param_list = local_info[0]["0.matmul1_weight"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == ((2, 0, 3, 1), -1)

    # specific rank
    rank_info = get_strategy_metadata(network=train_net, rank_id=6)
    param_list = rank_info[6]["weight0"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == (-1, -1)

    # global rank
    global_info = get_strategy_metadata(network=train_net)
    param_list = global_info[10]["1.matmul2_weight"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == ((2, 0, 3, 1), -1)


def test_stand_alone():
    """
    Feature: Get params in stand alone mode.
    Description: stand alone mode.
    Expectation: compile success.
    """
    import math
    import mindspore as ms
    from mindspore.common.initializer import initializer, HeUniform
    from mindspore.parallel.auto_parallel import AutoParallel

    class Network(nn.Cell):
        def __init__(self, strategy=None):
            super().__init__()
            self.flatten = ops.Flatten()
            weight_init = HeUniform(math.sqrt(5))
            self.fc1_weight = Parameter(initializer(weight_init, [28 * 28, 512], ms.dtype.float32), name="fc1")
            self.fc2_weight = Parameter(initializer(weight_init, [512, 512], ms.dtype.float32), name="fc2")
            self.fc3_weight = Parameter(initializer(weight_init, [512, 10], ms.dtype.float32), name="fc3")
            self.matmul1 = ops.MatMul()
            if strategy is not None:
                self.matmul1.shard(in_strategy=strategy)
            self.relu1 = ops.ReLU()
            self.matmul2 = ops.MatMul()
            self.relu2 = ops.ReLU()
            self.matmul3 = ops.MatMul()

        def construct(self, x):
            x = self.flatten(x)
            x = self.matmul1(x, self.fc1_weight)
            x = self.relu1(x)
            x = self.matmul2(x, self.fc2_weight)
            x = self.relu2(x)
            logits = self.matmul3(x, self.fc3_weight)
            return logits

    standalone_dataset = FakeData(size=8, batch_size=8, image_size=(28, 28), num_classes=10)

    # net
    with no_init_parameters():
        standalone_net = Network(strategy=None)
        net_optim = nn.Momentum(standalone_net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # set mode
    init_hccl(global_rank=0, device_num=1)
    enable_save_strategy_online()
    network = AutoParallel(standalone_net)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
    model = Model(network=network, loss_fn=loss_fn, optimizer=net_optim)
    model.train(epoch=2, train_dataset=standalone_dataset, callbacks=None, dataset_sink_mode=False)

    # net info
    local_info = get_current_strategy_metadata(network=network)
    assert local_info is None

    global_layout = get_strategy_metadata(network=network)
    assert global_layout is None
