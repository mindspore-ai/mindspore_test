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

import os
import pytest
import numpy as np

from mindspore import context, ops
from mindspore import log as logger
import mindspore.nn as nn
from mindspore.nn.utils import no_init_parameters
from mindspore.common import Tensor, Parameter, set_seed
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, load_checkpoint, load_param_into_net, \
    get_ckpt_path_with_strategy
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.parallel.auto_parallel import AutoParallel


context.set_context(mode=context.GRAPH_MODE)
init()


# creat dataset
class FakeData:
    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224), num_classes=10, random_offset=0,
                 use_parallel=False):
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.num_classes = num_classes

        self.batch_index = 0
        self.image_data_type = np.float32
        self.label_data_type = np.float32
        self.is_onehot = True

        self.rank_size = 1
        self.rank_id = 0

        if use_parallel is True:
            self.rank_size = get_group_size()
            self.rank_id = get_rank()

        self.total_batch_size = self.rank_batch_size * self.rank_size
        self.total_batch_data_size = (self.rank_size, self.rank_batch_size) + image_size

    def get_dataset_size(self):
        return int(self.size / self.total_batch_size)

    def create_tuple_iterator(self, num_epochs=-1, do_copy=False):
        return self

    def __getitem__(self, batch_index):
        rng_state = np.random.get_state()
        np.random.seed(batch_index + self.random_offset)
        img = np.random.randn(*self.total_batch_data_size)
        target = np.random.randint(0, self.num_classes, size=(self.rank_size, self.rank_batch_size))

        np.random.set_state(rng_state)
        img = img[self.rank_id]
        target = target[self.rank_id]
        img_ret = img.astype(self.image_data_type)
        target_ret = target.astype(self.label_data_type)

        if self.is_onehot:
            target_onehot = np.zeros(shape=(self.rank_batch_size, self.num_classes))
            target_onehot[np.arange(self.rank_batch_size), target] = 1
            target_ret = target_onehot.astype(self.label_data_type)
        return Tensor(img_ret), Tensor(target_ret)

    def __len__(self):
        return self.size

    def __iter__(self):
        self.batch_index = 0
        return self

    def reset(self):
        self.batch_index = 0

    def __next__(self):
        if self.batch_index * self.total_batch_size < len(self):
            data = self[self.batch_index]
            self.batch_index += 1
            return data
        raise StopIteration


class Network(nn.Cell):
    def __init__(self, strategy=None):
        super().__init__()
        self.flatten = ops.Flatten()
        self.fc1_weight = Parameter(Tensor(np.full([28 * 28, 512], 0.01, np.float32)), name="fc1")
        self.fc2_weight = Parameter(Tensor(np.full([512, 512], 0.01, np.float32)), name="fc2")
        self.fc3_weight = Parameter(Tensor(np.full([512, 10], 0.01, np.float32)), name="fc3")
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


# set auto_parallel
def set_parallel_mode(obj, parallel_config=None):
    if parallel_config is None:
        return obj
    parallel_mode = parallel_config.get("parallel_mode", "semi_auto")
    net = AutoParallel(obj, parallel_mode)
    if parallel_config.get("dataset_strategy", None) is not None:
        net.dataset_strategy(parallel_config["dataset_strategy"])
    if parallel_config.get("pipeline_stages", None) is not None:
        net.pipeline(parallel_config["pipeline_stages"])
    if parallel_config.get("save_strategy_file", None) is not None:
        net.save_param_strategy_file(parallel_config["save_strategy_file"])
    if parallel_config.get("load_strategy_file", None) is not None:
        net.load_param_strategy_file(parallel_config["load_strategy_file"])
    if parallel_config.get("enable_parallel_optimizer", None) is True:
        net.hsdp()
    return net


# clean ckpt
def clean_all_ckpt_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                try:
                    os.remove(os.path.join(folder_path, file_name))
                except FileNotFoundError as e:
                    logger.warning("[{}] remove ckpt file error.".format(e))


# find ckpt
def find_newest_ckpt_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith('.ckpt'),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


# creat model
def create_train_model(network, parallel_config):
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
    net_optim = nn.Momentum(network.trainable_params(), learning_rate=0.01, momentum=0.9)
    network = nn.WithLossCell(network, loss_fn)
    network = nn.PipelineCell(network, 4)
    network = set_parallel_mode(network, parallel_config)
    model = Model(network, optimizer=net_optim)
    return model


# save checkpoint when model train
def model_train(model, epoch, dataset, ckpt_path, ckpt_prefix, integral_save, remove_redundancy):
    model.build(train_dataset=dataset, epoch=epoch)
    model.train_network.init_parameters_data(auto_parallel_mode=True)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=1,
                                   keep_checkpoint_max=5,
                                   integrated_save=integral_save,
                                   async_save=False,
                                   remove_redundancy=remove_redundancy)
    ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)
    clean_all_ckpt_files(ckpt_path)
    model.train(epoch=epoch, train_dataset=dataset, callbacks=[ckpt_callback])


# Find the available checkpoint file and return the paths.
def parallel_mode_get_ckpt_path_with_strategy(strategy_file=None, cpkt_path=None):
    ckpt_file = find_newest_ckpt_file(cpkt_path)
    strategy_file = strategy_file
    ckpt_file_new = get_ckpt_path_with_strategy(ckpt_file, strategy_file)
    print(f"ckpt_file_new {ckpt_file_new}")


# load the newest checkpoint and predict
def load_newest_cpkt_predict(model, parallel_config, ckpt_path, remove_redundancy, inputs, label=None):
    newest_ckpt_file = find_newest_ckpt_file(ckpt_path)
    ckpt_path = ckpt_path
    param_dict = load_checkpoint(newest_ckpt_file, remove_redundancy=remove_redundancy)
    param_not_load, _ = load_param_into_net(model.train_network, param_dict, remove_redundancy=remove_redundancy)
    print(f"param_not_load = {param_not_load}")

    set_parallel_mode(model.train_network, parallel_config)
    predict_result = model.predict(inputs, label)
    return predict_result


# compare accuracy
def compare_params(ex_params, actual_params):
    assert np.allclose(ex_params, actual_params, atol=1e-3, rtol=1e-3)


def semi_auto_ckpt_with_redundancy_dp2_mp2_pp2():
    cur_dir = './test_cpkt_pp2/auto_parallel/with_redundancy_init'
    context.set_context(save_graphs=True, save_graphs_path=f'{cur_dir}/graphs')
    set_seed(12)
    parallel_dataset = FakeData(size=8 * 4, batch_size=8 * 4, image_size=(28, 28), num_classes=10)
    inputs = Tensor(np.random.randn(8, 28, 28).astype(np.float32))
    label = Tensor(np.random.randn(8, 10).astype(np.float32))

    # net
    in_strategy = ((1, 2), (2, 2))
    with no_init_parameters():
        parallel_net = Network1(strategy=in_strategy)
    parallel_net.flatten.pipeline_stage = 0
    parallel_net.layer1.pipeline_stage = 0
    parallel_net.relu1.pipeline_stage = 0
    parallel_net.layer2.pipeline_stage = 0
    parallel_net.relu2.pipeline_stage = 1
    parallel_net.layer3.pipeline_stage = 1

    # creat model
    stra_ckpt_file = f"{cur_dir}/train_strategy.ckpt"
    parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel", "pipeline_stages": 2,
                       "save_strategy_file": stra_ckpt_file}
    parallel_model = create_train_model(parallel_net, parallel_config)

    # model train
    remove_redundancy = False
    ckpt_path = f"{cur_dir}/rank_{get_rank()}_ckpt"
    model_train(model=parallel_model, epoch=2, dataset=parallel_dataset, ckpt_path=ckpt_path,
                ckpt_prefix="ckpt_parallel_wd", integral_save=False, remove_redundancy=remove_redundancy)

    # find ckpt file
    parallel_mode_get_ckpt_path_with_strategy(strategy_file=stra_ckpt_file, cpkt_path=ckpt_path)

    # predict, remove_redundancy=False
    parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel", "pipeline_stages": 2,
                       "load_strategy_file": stra_ckpt_file}
    my_predict = load_newest_cpkt_predict(parallel_model, parallel_config, ckpt_path, remove_redundancy, inputs, label)
    print(f"with_redundancy_predict {my_predict}")
    return my_predict


def semi_auto_ckpt_remove_redundancy_dp2_mp2_pp2():
    cur_dir = './test_cpkt_pp2/auto_parallel/remove_redundancy_init'
    context.set_context(save_graphs=True, save_graphs_path=f'{cur_dir}/graphs')
    set_seed(12)
    parallel_dataset = FakeData(size=8 * 4, batch_size=8 * 4, image_size=(28, 28), num_classes=10)
    inputs = Tensor(np.random.randn(8, 28, 28).astype(np.float32))
    label = Tensor(np.random.randn(8, 10).astype(np.float32))

    # net
    in_strategy = ((1, 2), (2, 2))
    with no_init_parameters():
        parallel_net = Network1(strategy=in_strategy)
    parallel_net.flatten.pipeline_stage = 0
    parallel_net.layer1.pipeline_stage = 0
    parallel_net.relu1.pipeline_stage = 0
    parallel_net.layer2.pipeline_stage = 0
    parallel_net.relu2.pipeline_stage = 1
    parallel_net.layer3.pipeline_stage = 1

    # creat model
    stra_ckpt_file = f"{cur_dir}/train_strategy.ckpt"
    parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel", "pipeline_stages": 2,
                       "save_strategy_file": stra_ckpt_file}
    parallel_model = create_train_model(parallel_net, parallel_config)

    # model train
    remove_redundancy = True
    ckpt_path = f"{cur_dir}/rank_{get_rank()}_ckpt"
    model_train(model=parallel_model, epoch=2, dataset=parallel_dataset, ckpt_path=ckpt_path,
                ckpt_prefix="ckpt_parallel_wd", integral_save=False, remove_redundancy=remove_redundancy)

    # find ckpt file
    parallel_mode_get_ckpt_path_with_strategy(strategy_file=stra_ckpt_file, cpkt_path=ckpt_path)

    # predict, remove_redundancy=True
    parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel", "pipeline_stages": 2,
                       "load_strategy_file": stra_ckpt_file}
    my_predict = load_newest_cpkt_predict(parallel_model, parallel_config, ckpt_path, remove_redundancy, inputs, label)
    print(f"remove_redundancy_predict {my_predict}")
    return my_predict


def test_remove_redundancy_init_transfer_precision():
    """
    Feature: save_checkpoints and load_checkpoints with remove redundancy or not.
    Description: net with strategy in auto parallel mode, compare predict accuracy.
    Expectation: the predict results meets requirement.
    """
    ex_predict = semi_auto_ckpt_with_redundancy_dp2_mp2_pp2()
    rm_predict = semi_auto_ckpt_remove_redundancy_dp2_mp2_pp2()
    compare_params(ex_predict, rm_predict)


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

    # set mode
    standalone_net = AutoParallel(standalone_net)
    # disable pylint too broad Exception
    # pylint: disable=W0212
    standalone_net._device_num = 1

    # creat model
    remove_redundancy = True
    parallel_config = {}
    standalone_model = create_train_model(standalone_net, parallel_config)

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
