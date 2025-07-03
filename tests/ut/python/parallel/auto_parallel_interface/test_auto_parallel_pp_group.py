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
from mindspore.nn import L1Loss, PipelineCell
from mindspore.nn.utils import no_init_parameters
from mindspore.train import Model
from parallel.auto_parallel_interface._utils_dataset import FakeData
from parallel.auto_parallel_interface._utils import init_hccl, set_parallel_mode, save_ir_graphs, check_node_attrs_pair, \
    find_ir_file_path


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


def test_parallel_optimizer_pp2dp4_zero3_strategy_pp2_full_op():
    """
    Feature: shard with stategy, full shard.
    Description: send and receive op share group.
    Expectation: compile success, validation pass.
    """
    ir_file_path = save_ir_graphs("test_parallel_optimizer_pp2dp4_zero3_strategy_pp2_full_op")
    np.random.seed(100)
    weight1 = Tensor(0.1 * np.random.randn(96, 16).astype(np.float32))
    weight2 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight3 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight4 = Tensor(0.1 * np.random.randn(16, 16).astype(np.float32))
    weight_list = [[weight1, weight2], [weight3, weight4]]

    # set auto_parallel
    init_hccl(global_rank=0, device_num=16)
    optim_cfg = {"optimizer_level": "level3", "optimizer_weight_shard_size": -1, "parallel_optimizer_threshold": 0}
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

    # validate send-receive group
    validate_ir = find_ir_file_path(ir_file_path, "validate")
    check_pairs = {"Send(%5)": {"2-16557109384257890687Send": 1}}
    check_node_attrs_pair(validate_ir, check_pairs)
    check_pairs = {"Receive(%13)": {"2-16557109384257890687Send": 1}}
    check_node_attrs_pair(validate_ir, check_pairs)
