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

import math
import pytest
import numpy as np

import mindspore as ms
import mindspore.nn as nn
from mindspore import context, ops
from mindspore.common import Tensor, Parameter, set_seed
from mindspore.nn.utils import no_init_parameters
from mindspore.common.initializer import initializer, HeUniform
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, load_checkpoint, load_param_into_net, LossMonitor
from mindspore.communication.management import init, get_rank
from mindspore.parallel.auto_parallel import AutoParallel
from tests.st.auto_parallel.utils.dataset_utils import FakeData
from tests.st.auto_parallel.utils._utils import set_parallel_mode, clean_all_ckpt_files, find_newest_ckpt_file, \
    compare_params, clear_files_in_directory, parallel_mode_get_ckpt_path_with_strategy


context.set_context(mode=context.GRAPH_MODE)
init()


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


class Network1(nn.Cell):
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


# creat model
def create_train_model(network, optimizer, parallel_config):
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
    network = nn.WithLossCell(network, loss_fn)
    network = nn.PipelineCell(network, 4)
    network = set_parallel_mode(network, parallel_config)
    model = Model(network, optimizer=optimizer)
    return model


# save checkpoint when model train
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


# load the newest checkpoint and predict
def load_newest_cpkt_predict(model, parallel_config, ckpt_path, remove_redundancy, inputs, label=None):
    newest_ckpt_file = find_newest_ckpt_file(ckpt_path)
    param_dict = load_checkpoint(newest_ckpt_file, remove_redundancy=remove_redundancy)
    param_not_load, _ = load_param_into_net(model.train_network, param_dict, remove_redundancy=remove_redundancy)
    print(f"param_not_load = {param_not_load}")

    set_parallel_mode(model.train_network, parallel_config)
    predict_result = model.predict(inputs, label)
    return predict_result


def semi_auto_ckpt_redundancy_dp2_mp2_pp2(cur_root_dir, remove_redundancy):
    # save ir graph
    ir_graph_path = f"{cur_root_dir}/graphs"
    clear_files_in_directory(f"{ir_graph_path}/rank_{get_rank()}")
    context.set_context(save_graphs=True, save_graphs_path=ir_graph_path)

    # dataset
    set_seed(1)
    parallel_dataset = FakeData(size=8 * 4, batch_size=8 * 4, image_size=(28, 28), num_classes=10)
    inputs = Tensor(np.random.randn(8, 28, 28).astype(np.float32))
    label = Tensor(np.random.randn(8, 10).astype(np.float32))

    # net
    in_strategy = ((1, 2), (2, 2))
    with no_init_parameters():
        parallel_net = Network1(strategy=in_strategy)
        net_optim = nn.Momentum(parallel_net.trainable_params(), learning_rate=0.01, momentum=0.9)
    parallel_net.flatten.pipeline_stage = 0
    parallel_net.layer1.pipeline_stage = 0
    parallel_net.relu1.pipeline_stage = 0
    parallel_net.layer2.pipeline_stage = 0
    parallel_net.relu2.pipeline_stage = 1
    parallel_net.layer3.pipeline_stage = 1

    # creat model
    stra_ckpt_file = f"{cur_root_dir}/train_strategy.ckpt"
    parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel", "pipeline_stages": 2,
                       "save_strategy_file": stra_ckpt_file}
    parallel_model = create_train_model(parallel_net, net_optim, parallel_config)

    # model train
    ckpt_path = f"{cur_root_dir}/rank_{get_rank()}_ckpt"
    model_train(model=parallel_model, epoch=2, dataset=parallel_dataset, ckpt_path=ckpt_path,
                ckpt_prefix="ckpt_parallel_wd", integral_save=False, remove_redundancy=remove_redundancy)

    # find checkpoint file
    parallel_mode_get_ckpt_path_with_strategy(strategy_file=stra_ckpt_file, cpkt_path=ckpt_path)

    # predict
    parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel", "pipeline_stages": 2,
                       "load_strategy_file": stra_ckpt_file}
    my_predict = load_newest_cpkt_predict(parallel_model, parallel_config, ckpt_path, remove_redundancy, inputs, label)
    print(f"with_redundancy_predict {my_predict}")
    return my_predict


def test_cpkt_remove_redundancy_precision():
    """
    Feature: save_checkpoints and load_checkpoints with remove redundancy or not.
    Description: net with strategy in auto parallel mode, compare predict accuracy.
    Expectation: the predict results meets requirement.
    """
    # case 1, save param in checkpoints file full in all rank
    with_redundancy_root_path = "./test_cpkt_pp2/auto_parallel/with_redundancy_init"
    with_redundancy = False
    with_redundancy_predict = semi_auto_ckpt_redundancy_dp2_mp2_pp2(with_redundancy_root_path, with_redundancy)
    print("with_redundancy_predict_result", with_redundancy_predict)

    # case 2, remove redundancy param in checkpoints file in each rank
    remove_redundancy_root_path = "./test_cpkt_pp2/auto_parallel/remove_redundancy_init"
    remove_redundancy = True
    remove_redundancy_predict = semi_auto_ckpt_redundancy_dp2_mp2_pp2(remove_redundancy_root_path, remove_redundancy)
    print("remove_redundancy_predict_result", remove_redundancy_predict)

    # compare accuracy with case 1 and case 2
    compare_params(with_redundancy_predict, remove_redundancy_predict)


def test_stand_alone_remove_redundancy():
    """
    Feature: remove_redundancy in stand alone mode.
    Description: integral_save is True, remove_redundancy is True.
    Expectation: raise ValueError.
    """
    standalone_dataset = FakeData(size=8, batch_size=8, image_size=(28, 28), num_classes=10)
    predict_inputs = Tensor(np.random.randn(8, 28, 28).astype(np.float32))

    # net
    with no_init_parameters():
        standalone_net = Network(strategy=None)
        net_optim = nn.Momentum(standalone_net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # set mode
    standalone_net = AutoParallel(standalone_net)
    # disable pylint too broad Exception
    # pylint: disable=W0212
    standalone_net._device_num = 1

    # creat model
    remove_redundancy = True
    parallel_config = None
    standalone_model = create_train_model(standalone_net, net_optim, parallel_config)

    # train
    ckpt_path = "./test_cpkt_pp2/auto_parallel/stand_alone_init/rank_{}_ckpt".format(get_rank())
    model_train(standalone_model, epoch=2, dataset=standalone_dataset,
                ckpt_path=ckpt_path, ckpt_prefix="ckpt_standalone",
                integral_save=True, remove_redundancy=remove_redundancy)

    # predict
    with pytest.raises(ValueError):
        newest_ckpt_file = find_newest_ckpt_file(ckpt_path)
        param_dict = load_checkpoint(ckpt_file_name=newest_ckpt_file, remove_redundancy=remove_redundancy)
        load_param_into_net(net=standalone_net, parameter_dict=param_dict, remove_redundancy=remove_redundancy)
        standalone_model.predict(predict_inputs)
