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

import mindspore.nn as nn
from mindspore import context
from mindspore._c_expression import GroupOptions
from mindspore.common import set_seed, lazy_inline
from mindspore.nn.utils import no_init_parameters
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, LossMonitor
from mindspore.communication.management import init, get_rank, create_group, destroy_group
from mindspore.parallel.strategy import get_strategy_metadata, enable_save_strategy_online
from tests.st.auto_parallel.utils.dataset_utils import FakeData
from tests.st.auto_parallel.utils._utils import set_parallel_mode, clean_all_ckpt_files, save_ir_graphs

context.set_context(mode=context.GRAPH_MODE)
init()


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


class Network(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        self.flatten = nn.Flatten()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.layer1 = nn.Dense(28 * 28, 512)
        if strategy is not None:
            self.layer1.matmul.shard(in_strategy=strategy)
        self.layer2 = nn.Dense(512, 512)
        self.layer3 = nn.Dense(512, 10)

    def construct(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        logits = self.layer3(x)
        return logits


def create_pp_train_model(network, micro_size, stage_config, parallel_config, optimizer):
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
    network = WithLossCell(network, loss_fn)
    network = nn.PipelineCell(network, micro_size, stage_config)
    network = set_parallel_mode(network, parallel_config)
    model = Model(network, optimizer=optimizer)
    return model


def model_train(model, epoch, dataset, ckpt_path, ckpt_prefix, integral_save, remove_redundancy):
    ckpt_config = CheckpointConfig(save_checkpoint_steps=1,
                                   keep_checkpoint_max=5,
                                   integrated_save=integral_save,
                                   async_save=False,
                                   remove_redundancy=remove_redundancy)
    loss_cb = LossMonitor(1)
    ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)
    clean_all_ckpt_files(ckpt_path)
    model.train(epoch=epoch, train_dataset=dataset, callbacks=[ckpt_callback, loss_cb], dataset_sink_mode=False)


def semi_auto_dp2_mp2_pp2(cur_root_dir, remove_redundancy=False, opt_config=None):
    # save ir graph
    ir_graph_path = f"{cur_root_dir}/graphs"
    save_ir_graphs(ir_graph_path)

    # dataset
    set_seed(1)
    parallel_dataset = FakeData(size=8 * 4, batch_size=8 * 4, image_size=(28, 28), num_classes=10)

    # net
    in_strategy = ((1, 2), (2, 2))
    with no_init_parameters():
        parallel_net = Network(strategy=in_strategy)
        net_optim = nn.Momentum(parallel_net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # creat model
    stra_ckpt_file = f"{cur_root_dir}/train_strategy.ckpt"
    pp_config = {"stages": 2}
    parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel", "pipeline_config": pp_config,
                       "save_strategy_file": stra_ckpt_file, "parallel_optimizer_config": opt_config}
    stage_config = {"_backbone.flatten": 0, "_backbone.layer1": 0, "_backbone.relu1": 0, "_backbone.layer2": 0,
                    "_backbone.relu2": 1, "_backbone.layer3": 1}
    parallel_model = create_pp_train_model(parallel_net, 4, stage_config, parallel_config, net_optim)

    # model train
    ckpt_path = f"{cur_root_dir}/rank_{get_rank()}_ckpt"
    model_train(model=parallel_model, epoch=2, dataset=parallel_dataset, ckpt_path=ckpt_path,
                ckpt_prefix="ckpt_parallel_wd", integral_save=False, remove_redundancy=remove_redundancy)
    return parallel_model


def test_hccl_config():
    """
    Feature: hccl buffle size, get strategy.
    Description: customized value of specific groups.
    Expectation: run success.
    """
    enable_save_strategy_online()

    # Zero1
    opt_config = {"optimizer_level": "level1", "optimizer_weight_shard_size": 2, "parallel_optimizer_threshold": 0}
    opt_parallel_model = semi_auto_dp2_mp2_pp2(cur_root_dir="./test_hccl_config/zero1", opt_config=opt_config)
    opt_train_info = get_strategy_metadata(opt_parallel_model.train_network)
    opt_param_list = opt_train_info[1]["layer2.weight"]
    opt_param_layout = opt_param_list[0].to_dict()
    assert opt_param_layout['tensor_map'] == (0, -1)

    # do not enable optimizer parallel
    opt_config = None
    parallel_model = semi_auto_dp2_mp2_pp2(cur_root_dir="./test_hccl_config/no_opt", opt_config=opt_config)
    train_info = get_strategy_metadata(parallel_model.train_network)
    param_list = train_info[1]["layer2.weight"]
    param_layout = param_list[0].to_dict()
    assert param_layout['tensor_map'] == (-1, -1)


def test_union_hccl_groups():
    """
    Feature: Union hccl groups creation.
    Description: If hccl groups has been created in python by create_group, it will not be created repeatedly in C++.
    Expectation: run success.
    """
    group0 = "customed groups 0-1, 0"
    options0 = GroupOptions()
    options0.hccl_config = {}

    group1 = "customed groups 0-1, 1"
    options1 = GroupOptions()
    options1.hccl_config = {"hccl_buffer_size": 400}

    group2 = "customed groups 0-1, 2"
    options2 = GroupOptions()
    options2.hccl_config = {"hccl_buffer_size": 100}

    group3 = "customed groups 0-1, 3"
    options3 = GroupOptions()
    options3.hccl_config = {"hccl_buffer_size": 200}

    # case 1, create_group, destroy before parallel, create default
    epoch = 0
    rank_ids = [0, 1]
    if get_rank() in rank_ids:
        create_group(group0, rank_ids, options0)
        destroy_group(group0)
    semi_auto_dp2_mp2_pp2(cur_root_dir=f"./test_union_hccl_groups/run_{epoch}", opt_config=None)

    # case 2, create_group, reused in parallel
    epoch += 1
    rank_ids = [0, 1]
    if get_rank() in rank_ids:
        create_group(group1, rank_ids, options1)
        create_group(group2, rank_ids, options2)
    semi_auto_dp2_mp2_pp2(cur_root_dir=f"./test_union_hccl_groups/run_{epoch}", opt_config=None)

    # case 3, destroy group, create default
    epoch += 1
    rank_ids = [0, 1]
    if get_rank() in rank_ids:
        destroy_group(group1)
        destroy_group(group2)
    semi_auto_dp2_mp2_pp2(cur_root_dir=f"./test_union_hccl_groups/run_{epoch}", opt_config=None)

    # case 4, create group again, reused in parallel
    epoch += 1
    rank_ids = [0, 1]
    if get_rank() in rank_ids:
        create_group(group3, rank_ids, options3)
    semi_auto_dp2_mp2_pp2(cur_root_dir=f"./test_union_hccl_groups/run_{epoch}", opt_config=None)
