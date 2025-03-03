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

import copy
import json
import os
import time
import numpy as np

import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import lazy_inline
from mindspore import context
from mindspore import log as logger
from mindspore.nn.utils import no_init_parameters
from mindspore.common import Tensor, set_seed
from mindspore.communication.management import init, get_rank
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, load_checkpoint, load_param_into_net
from mindspore.parallel.auto_parallel import AutoParallel
from tests.st.auto_parallel.utils.dataset_utils import FakeData
from tests.st.auto_parallel.utils._utils import set_parallel_mode, clean_all_ckpt_files, compare_nparray, \
    clear_files_in_directory, find_newest_ckpt_file_by_name

context.set_context(mode=context.GRAPH_MODE)
init()


class LeNet5WithCell(nn.Cell):
    @lazy_inline
    def __init__(self, pipeline_flag=False, strategy=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, pad_mode='valid',
                               weight_init='normal')
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, pad_mode='valid',
                               weight_init='normal')
        self.fc1 = nn.Dense(in_channels=16 * 5 * 5, out_channels=120, weight_init='normal',
                            bias_init='zeros')
        self.fc2 = nn.Dense(in_channels=120, out_channels=84, weight_init='normal',
                            bias_init='zeros')
        self.fc3 = nn.Dense(in_channels=84, out_channels=10, weight_init='normal',
                            bias_init='zeros')
        self.relu = P.ReLU()
        self.relu1 = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2, pad_mode='valid')
        self.flatten = nn.Flatten()
        if strategy is not None:
            self.relu.shard(strategy)
        if pipeline_flag is True:
            self.conv1.pipeline_stage = 0
            self.conv2.pipeline_stage = 0
            self.fc1.pipeline_stage = 1
            self.fc2.pipeline_stage = 1
            self.fc3.pipeline_stage = 1
            self.relu.pipeline_stage = 0
            self.max_pool2d.pipeline_stage = 0
            self.flatten.pipeline_stage = 0

    def construct(self, x, label):
        # input shape: n*1*32*32, out shape: n*6*28*28
        x = self.conv1(x)
        x = self.relu(x)
        # input shape: n*6*28*28, out shape: n*6*14*14
        x = self.max_pool2d(x)
        # input shape: n*6*14*14, out shape: n*16*10*10
        x = self.conv2(x)
        x = self.relu1(x)
        # input shape: n*16*10*10, out shape: n*16*5*5
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.fc3(x)
        return x


# parallel_speed_up_json_config of computation_communication_fusion_level
def set_ascend_config(file_path, key, value):
    if not os.path.exists(file_path):
        cur_json_dir = os.path.dirname(file_path)
        os.makedirs(cur_json_dir, exist_ok=True)
        clear_files_in_directory(cur_json_dir)
        speed_up_json_config = {key: value}
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(speed_up_json_config, f, ensure_ascii=False, indent=4)
        print(f"Set ascend config {key} to {value}.")
        time.sleep(5)
    else:
        print(f"File {file_path} already exists. Skipping the operation.")


# stand_alone mode
def set_stand_alone_mode(net):
    net = AutoParallel(net)
    # disable pylint too broad Exception
    # pylint: disable=W0212
    net._device_num = 1
    return net


# creat model
def create_train_model(network, optimizer, parallel_config):
    # loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction='mean')
    loss_fn = None
    network = set_parallel_mode(network, parallel_config)
    model = Model(network=network, loss_fn=loss_fn, optimizer=optimizer)
    return model


# save checkpoint when model train
def model_train(model, epoch, dataset, ckpt_path, ckpt_prefix, integral_save, remove_redundancy):
    ckpt_config = CheckpointConfig(save_checkpoint_steps=1,
                                   keep_checkpoint_max=5,
                                   integrated_save=integral_save,
                                   async_save=False,
                                   remove_redundancy=remove_redundancy)
    ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_path, config=ckpt_config)
    logger.info(f"clean all Checkpoint file under {ckpt_path}")
    clean_all_ckpt_files(ckpt_path)
    model.train(epoch=epoch, train_dataset=dataset, callbacks=[ckpt_callback], dataset_sink_mode=False)


# load checkpints files and predict
def load_cpkt_model_predict(net, parallel_config, checkpoints_dir, *inputs):
    net = set_parallel_mode(net, parallel_config)

    # load param from checkpoints file to net
    newest_checkpoint_file = find_newest_ckpt_file_by_name(checkpoints_dir)
    param_dict = load_checkpoint(newest_checkpoint_file)
    param_not_load, _ = load_param_into_net(net, param_dict)
    print("param not load is ", param_not_load, flush=True)

    # predict net
    model = Model(net)
    predict_result = model.predict(*inputs)
    return predict_result


def stand_alone_train(cur_root_dir, checkpoints_dir):
    # stand_alone not set computation_communication_fusion_level
    rank_id = get_rank()
    set_seed(1)

    # dataset
    standalone_dataset = FakeData(size=256, batch_size=128, image_size=(1, 32, 32))

    # save ir graph
    ir_graph_path = f"{cur_root_dir}/graphs"
    clear_files_in_directory(f"{ir_graph_path}/rank_{rank_id}")
    context.set_context(save_graphs=2, save_graphs_path=ir_graph_path)

    # delay init net and opt param
    with no_init_parameters():
        train_net = LeNet5WithCell()
        net_optim = nn.Momentum(params=train_net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # creat train model and set parallel mode
    train_net = set_stand_alone_mode(train_net)
    parallel_config = None
    standalone_model = create_train_model(train_net, net_optim, parallel_config)

    # train net with model
    model_train(model=standalone_model, epoch=5, dataset=standalone_dataset, ckpt_path=checkpoints_dir,
                ckpt_prefix="ckpt_standalone", integral_save=True, remove_redundancy=False)


def parallel_mode_train(cur_root_dir, checkpoints_dir):
    rank_id = get_rank()
    set_seed(1)

    # parallel dataset
    parallel_dataset = FakeData(size=256, batch_size=16, image_size=(1, 32, 32))

    # save ir graph
    ir_graph_path = f"{cur_root_dir}/graphs"
    clear_files_in_directory(f"{ir_graph_path}/rank_{rank_id}")
    context.set_context(save_graphs=2, save_graphs_path=ir_graph_path)

    # delay init net and opt param
    with no_init_parameters():
        train_net = nn.PipelineCell(LeNet5WithCell(pipeline_flag=True), 4)
        net_optim = nn.Momentum(params=train_net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # creat model, set semi_auto_parallel and ascend_config
    train_stra_ckpt_file = f"{cur_root_dir}/train_strategy.ckpt"
    speed_up_json_path = f"{cur_root_dir}/ascend_config/parallel_speed_up.json"
    set_ascend_config(speed_up_json_path, "enable_begin_end_inline_opt", True)
    parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel",
                       "save_strategy_file": train_stra_ckpt_file, "ascend_config": speed_up_json_path,
                       "enable_parallel_optimizer": True}
    parallel_model = create_train_model(train_net, net_optim, parallel_config)

    # train net with model
    model_train(model=parallel_model, epoch=5, dataset=parallel_dataset, ckpt_path=checkpoints_dir,
                ckpt_prefix="ckpt_parallel", integral_save=True, remove_redundancy=False)


def test_ascend_config_enable_begin_end_inline_opt():
    """
    Feature: set ascend config by transformer_opt of auto_parallel interface.
    Description: enable_begin_end_inline_opt is true in parallel_speed_up.json.
    Expectation: the predict results meets requirement.
    """
    rank_id = get_rank()

    # train net in stand_alone
    stand_alone_root_dir = "./test_ascend_config/stand_alone"
    checkpoint_dir_expected = f"{stand_alone_root_dir}/rank_{rank_id}_ckpt"
    stand_alone_train(stand_alone_root_dir, checkpoint_dir_expected)

    # train net in semi auto
    semi_auto_root_dir = "./test_ascend_config/semi_auto"
    checkpoint_dir_actual = f"{semi_auto_root_dir}/rank_{rank_id}_ckpt"
    parallel_mode_train(semi_auto_root_dir, checkpoint_dir_actual)

    # define predict net
    with no_init_parameters():
        expected_net = LeNet5WithCell()

    # stand_alone mode
    expected_net = set_stand_alone_mode(expected_net)
    actual_net = copy.deepcopy(expected_net)

    # predict dataset
    set_seed(1)
    inputs_np = Tensor(np.random.randn(128, 1, 32, 32).astype(np.float32))
    label = Tensor(np.random.randn(1, 1, 1, 1).astype(np.float32))

    # load param to predict net with cpkt of stand_alone train
    parallel_config = None
    expected_out = load_cpkt_model_predict(expected_net, parallel_config, checkpoint_dir_expected, inputs_np, label)
    print("expected_out is ", expected_out.asnumpy(), flush=True)

    # load param to predict net with cpkt of parallel train
    actual_out = load_cpkt_model_predict(actual_net, parallel_config, checkpoint_dir_actual, inputs_np, label)
    print("actual_out is ", actual_out.asnumpy(), flush=True)

    # compare accuracy
    compare_nparray(expected_out.asnumpy(), actual_out.asnumpy(), rtol=0.001, atol=0.001)
