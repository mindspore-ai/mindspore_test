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
import math
import numpy as np

import mindspore as ms
import mindspore.dataset as ds
from mindspore import context, nn, ops, jit
from mindspore import log as logger
from mindspore.communication import init
from mindspore.communication.management import get_rank, get_group_size
from mindspore.common import Parameter, set_seed
from mindspore.common.initializer import initializer, HeUniform
from mindspore.nn.utils import no_init_parameters
from mindspore.train import Model, CheckpointConfig, ModelCheckpoint, get_ckpt_path_with_strategy, LossMonitor
from mindspore.parallel.auto_parallel import AutoParallel


context.set_context(mode=context.GRAPH_MODE)
init()


# dataset
step_per_epoch = 4
var_single_batch_size = 16
var_in_dim = 32
var_hidden_dim = 8
var_out_dim = 16


def get_dataset(*input_data):
    def generate():
        for _ in range(step_per_epoch):
            yield (input_data,)

    return generate


# set auto_parallel
def set_parallel_mode(obj, parallel_config=None):
    if parallel_config is None:
        return obj
    parallel_mode = parallel_config.get("parallel_mode", "semi_auto")
    net = AutoParallel(obj, parallel_mode)
    if parallel_config.get("dataset_strategy", None) is not None:
        net.dataset_strategy(parallel_config["dataset_strategy"])
    if parallel_config.get("save_strategy_file", None) is not None:
        net.save_param_strategy_file(parallel_config["save_strategy_file"])
    if parallel_config.get("load_strategy_file", None) is not None:
        net.load_param_strategy_file(parallel_config["load_strategy_file"])
    return net


# find ckpt
def find_newest_ckpt_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith('.ckpt'),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


# clean ckpt
def clean_all_ckpt_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                try:
                    os.remove(os.path.join(folder_path, file_name))
                except FileNotFoundError as e:
                    logger.warning("[{}] remove ckpt file error.".format(e))


# Find the available checkpoint file and return the paths.
def parallel_mode_get_ckpt_path_with_strategy(strategy_file=None, cpkt_path=None):
    ckpt_file = find_newest_ckpt_file(cpkt_path)
    ckpt_path_with_strategy = get_ckpt_path_with_strategy(ckpt_file, strategy_file)
    print(f"ckpt_path_with_strategy {ckpt_path_with_strategy}")


# compare accuracy
def compare_params(ex_params, actual_params):
    assert np.allclose(ex_params.asnumpy(), actual_params.asnumpy(), atol=1e-3, rtol=1e-3)


# define net
class Net(nn.Cell):
    """define net"""

    def __init__(self, in_dim, hidden_dim, out_dim, strategy=None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        weight_init = HeUniform(math.sqrt(5))
        self.weight = Parameter(initializer(weight_init, [self.in_dim, self.hidden_dim]), "w")
        self.weight2 = Parameter(initializer(weight_init, [self.hidden_dim, self.out_dim]), "w2")

        self.matmul = ops.MatMul()
        if strategy is not None:
            self.matmul.shard(strategy)
        self.relu = ops.ReLU()
        self.matmul2 = ops.MatMul()

    def construct(self, x):
        out = self.matmul(x, self.weight)
        out = self.relu(out)
        out = self.matmul2(out, self.weight2)
        return out


def graph_model_train(parallel_config=None, ckpt_dirs=None, ckpt_prefix=None):
    # dataset
    set_seed(1)
    input_data = ms.Tensor(np.random.rand(var_single_batch_size, var_in_dim).astype(np.float32))
    label_data = ms.Tensor(np.random.rand(var_single_batch_size, var_out_dim).astype(np.float32))
    fake_dataset = get_dataset(input_data, label_data)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

    # net
    strategy = ((2, 4), (4, 1))
    with no_init_parameters():
        net = Net(var_in_dim, var_hidden_dim, var_out_dim, strategy)

    # creat model
    net_opt = nn.Momentum(net.trainable_params(), 0.01, 0.9)
    net_loss = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")
    net = set_parallel_mode(net, parallel_config)
    model = Model(network=net, loss_fn=net_loss, optimizer=net_opt)

    # initialize
    epoch = 2
    model.build(train_dataset=dataset, epoch=epoch)
    model.train_network.init_parameters_data(auto_parallel_mode=True)

    # model train
    clean_all_ckpt_files(ckpt_dirs)
    ckpt_config = CheckpointConfig(keep_checkpoint_max=1, integrated_save=False)
    loss_cb = LossMonitor(1)
    ckpt_callback = ModelCheckpoint(prefix=ckpt_prefix, directory=ckpt_dirs, config=ckpt_config)
    model.train(epoch=epoch, train_dataset=dataset, callbacks=[ckpt_callback, loss_cb], dataset_sink_mode=False)


def graph_model_predict(checkpoint_filenames=None, train_strategy_file=None, parallel_config=None):
    # predict dataset
    set_seed(1)
    predict_data = ms.Tensor(np.random.rand(var_single_batch_size, var_in_dim).astype(np.float32))

    # define predict net with delay init
    with no_init_parameters():
        net = Net(var_in_dim, var_hidden_dim, var_out_dim, strategy=None)

    # init data
    net.compile(predict_data)
    net = set_parallel_mode(net, parallel_config)

    # tansfer strategy
    model = ms.Model(net)
    predict_strategy = model.infer_predict_layout(predict_data)
    ms.load_distributed_checkpoint(net, checkpoint_filenames, predict_strategy, train_strategy_file)
    predict_result = model.predict(predict_data)
    return predict_result


# train and predict by model, set parallel by context
def cpkt_transfer_model_context():
    cur_dir = "./test_cpkt_transfer/model_context"
    context.set_context(save_graphs=True, save_graphs_path=f'{cur_dir}/graphs')

    # train
    context.reset_auto_parallel_context()
    strategy_ckpt_config = {"save_file": f"{cur_dir}/train_strategy.ckpt",
                            "only_trainable_params": True}
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8,
                                      strategy_ckpt_config=strategy_ckpt_config)
    ckpt_dirs = f"{cur_dir}/rank_{get_rank()}"
    ckpt_prefix = "ckpt_model_context"
    graph_model_train(parallel_config=None, ckpt_dirs=ckpt_dirs, ckpt_prefix=ckpt_prefix)
    context.reset_auto_parallel_context()

    # find cpkt file with train strategy
    parallel_mode_get_ckpt_path_with_strategy(strategy_ckpt_config["save_file"], ckpt_dirs)

    # predict
    train_strategy_file = strategy_ckpt_config["save_file"]
    context.set_auto_parallel_context(parallel_mode="semi_auto_parallel", device_num=8)
    checkpoint_filenames = [f"{cur_dir}/rank_{i}/{ckpt_prefix}-2_4.ckpt" for i in range(get_group_size())]
    my_predict = graph_model_predict(checkpoint_filenames, train_strategy_file)

    context.reset_auto_parallel_context()
    return my_predict


# train and predict by model, set parallel by auto_parallel
def cpkt_transfer_model_auto_parallel():
    cur_dir = "./test_cpkt_transfer/model_auto_parallel"
    context.set_context(save_graphs=True, save_graphs_path=f'{cur_dir}/graphs')

    # train
    stra_ckpt_file = f"{cur_dir}/train_strategy.ckpt"
    train_parallel_config = {"parallel_mode": "semi_auto", "save_strategy_file": stra_ckpt_file}
    ckpt_dirs = f"{cur_dir}/rank_{get_rank()}_ckpt"
    ckpt_prefix = "ckpt_model_auto_parallel"
    graph_model_train(parallel_config=train_parallel_config, ckpt_dirs=ckpt_dirs, ckpt_prefix=ckpt_prefix)

    # find cpkt file with train strategy
    parallel_mode_get_ckpt_path_with_strategy(stra_ckpt_file, ckpt_dirs)

    # predict
    train_strategy_file = stra_ckpt_file
    predict_parallel_config = {"parallel_mode": "semi_auto"}
    checkpoint_filenames = [f"{cur_dir}/rank_{i}_ckpt/{ckpt_prefix}-2_4.ckpt" for i in range(get_group_size())]
    my_predict = graph_model_predict(checkpoint_filenames, train_strategy_file, predict_parallel_config)
    return my_predict


def graph_func_train(parallel=True, parallel_config=None, cpkt_file=None):
    # dataset
    set_seed(1)
    input_data = ms.Tensor(np.random.rand(var_single_batch_size, var_in_dim).astype(np.float32))
    label_data = ms.Tensor(np.random.rand(var_single_batch_size, var_out_dim).astype(np.float32))
    fake_dataset = get_dataset(input_data, label_data)
    dataset = ds.GeneratorDataset(fake_dataset, ["input", "label"])

    # define shard
    if parallel:
        strategy = ((2, 4), (4, 1))

    # define train net with delay init
    with no_init_parameters():
        net = Net(var_in_dim, var_hidden_dim, var_out_dim, strategy)

    # define loss
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(reduction="mean")

    # define opt
    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # 1. define forward function
    def net_forward(x, y):
        out = net(x)
        loss = loss_fn(out, y)
        return loss

    # 2. Get gradient function
    grad_net = ops.value_and_grad(net_forward, grad_position=None, weights=net.trainable_params())

    # 3. Define function of one-step training
    @jit
    def train_one_step(inputs, target):
        loss, grads = grad_net(inputs, target)
        opt(grads)
        return loss

    # 4. set parallel mode
    parallel_net = set_parallel_mode(train_one_step, parallel_config)
    parallel_net.compile(input_data, label_data)
    parallel_net.init_parameters_data(auto_parallel_mode=True)
    opt.init_parameters_data(auto_parallel_mode=True)

    # 5. compile and init data, train net
    for _ in range(2):
        for input_x, label in dataset:
            loss = parallel_net(input_x, label)
            print("loss", loss, flush=True)

    # 6. save_checkpoint of net params
    params_list = net.trainable_params()
    ckpt_dir = os.path.dirname(cpkt_file)
    clean_all_ckpt_files(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(cpkt_file, 'w') as file:
        file.write("")
    ms.save_checkpoint(params_list, cpkt_file, integrated_save=False)


def graph_func_predict_net(predict_data, parallel_config=None):
    # define predict net with delay init
    with no_init_parameters():
        net = Net(var_in_dim, var_hidden_dim, var_out_dim, strategy=None)

    # transfer train_strategy.cpkt to predict_strategy.cpkt
    predict_net = set_parallel_mode(net, parallel_config)
    predict_net.compile(predict_data)
    return predict_net


def graph_func_predict_transform_checkpoint_by_rank(src_strategy_file, dst_strategy_file, src_checkpoints_files,
                                                    dst_checkpoints_files):
    rank_id = get_rank()
    ckpt_file = dst_checkpoints_files[rank_id]
    ckpt_dir = os.path.dirname(ckpt_file)
    os.makedirs(ckpt_dir, exist_ok=True)
    clean_all_ckpt_files(ckpt_dir)
    rank_list = ms.rank_list_for_transform(rank_id, src_strategy_file, dst_strategy_file)
    checkpoint_files_map = {}
    for rank in rank_list:
        checkpoint_files_map[rank] = src_checkpoints_files[rank]
    save_checkpoint_file_name = dst_checkpoints_files[rank_id]
    ms.transform_checkpoint_by_rank(rank_id, checkpoint_files_map, save_checkpoint_file_name, src_strategy_file,
                                    dst_strategy_file)


# functional train and predict, set parallel by context
def cpkt_transfer_func_auto_parallel():
    cur_dir = "./test_cpkt_transfer/functional_auto_parallel"
    context.set_context(save_graphs=True, save_graphs_path=f'{cur_dir}/graphs')

    # parallel train
    train_stra_ckpt_file = f"{cur_dir}/train_strategy.ckpt"
    train_parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel",
                             "save_strategy_file": train_stra_ckpt_file}

    train_ckpt_dirs = f"{cur_dir}/src_checkpoints_dir/rank_{get_rank()}"
    src_ckpt_file = f"{train_ckpt_dirs}/src_checkpoint_{get_rank()}.ckpt"
    graph_func_train(parallel=True, parallel_config=train_parallel_config, cpkt_file=src_ckpt_file)

    # find cpkt file with train strategy
    parallel_mode_get_ckpt_path_with_strategy(train_stra_ckpt_file, train_ckpt_dirs)

    # predict dataset
    set_seed(1)
    predict_data = ms.Tensor(np.random.rand(var_single_batch_size, var_in_dim).astype(np.float32))

    # parallel predict net
    predict_stra_ckpt_file = f"{cur_dir}/predict_strategy.ckpt"
    predict_parallel_config = {"parallel_mode": "semi_auto", "data_strategy": "data_parallel",
                               "save_strategy_file": predict_stra_ckpt_file}
    predict_net = graph_func_predict_net(predict_data, predict_parallel_config)

    # transfer1: load checkpoint_files and predict
    src_checkpoint_files = [f"{cur_dir}/src_checkpoints_dir/rank_{i}/src_checkpoint_{i}.ckpt" for i in
                            range(get_group_size())]
    train_strategy_file = train_stra_ckpt_file
    predict_strategy = predict_net.parameter_layout_dict
    ms.load_distributed_checkpoint(network=predict_net, checkpoint_filenames=src_checkpoint_files,
                                   predict_strategy=predict_strategy, train_strategy_filename=train_strategy_file)
    predict_result_load_distributed_checkpoint = predict_net(predict_data)

    # transfer2: rank_list_for_transform、transform_checkpoint_by_rank
    dst_checkpoint_files = [f"{cur_dir}/dst_checkpoints_dir/rank_{i}/dst_checkpoint_{i}.ckpt" for i in
                            range(get_group_size())]
    graph_func_predict_transform_checkpoint_by_rank(train_stra_ckpt_file, predict_stra_ckpt_file, src_checkpoint_files,
                                                    dst_checkpoint_files)
    param_dict = ms.load_checkpoint(dst_checkpoint_files[get_rank()], net=predict_net)
    param_not_load, _ = ms.load_param_into_net(net=predict_net, parameter_dict=param_dict)
    print("param_not_load in transform_checkpoints", param_not_load)
    predict_result_transform_checkpoints = predict_net(predict_data)

    # transfer3: load_segmented_checkpoints
    ckpt_file = dst_checkpoint_files[get_rank()]
    ckpt_dir = os.path.dirname(ckpt_file)
    param_dict_new = ms.load_segmented_checkpoints(ckpt_file_dir=ckpt_dir, net=predict_net)
    param_not_load, _ = ms.load_param_into_net(net=predict_net, parameter_dict=param_dict_new)
    print("param_not_load in load_segmented_checkpoints", param_not_load)
    predict_result_load_segmented_checkpoints = predict_net(predict_data)

    # result of transfer predict
    res = {"load_distributed_checkpoint": predict_result_load_distributed_checkpoint,
           "transform_checkpoints": predict_result_transform_checkpoints,
           "load_segmented_checkpoints": predict_result_load_segmented_checkpoints}
    return res


def test_parallel_functional_train_predict():
    """
    Feature: test save_checkpoints, load_checkpoints and transfer checkpoints.
    Description:
        1. define model train and predict, train net and save checkpoints with model, load_distributed_checkpoint to
        predict net.
        2. case 1 set parallel by context, train and predict net with model.
        3. case 2 set parallel by auto_parallel interface, train and predict net with model.
        3. define functional train and predict, set parallel by auto_parallel.
        4. case 3, load_distributed_checkpoint for predict net.
        5. case 4, transform_checkpoints, load_checkpoint and load_param_into_net to predict net.
        6. case 5, load_segmented_checkpoints and load_param_into_net to predict net.
        7. compare accuracy of predict result of case 2 ~ case 5 with case 1.
    Expectation:
        1.train and predict ok
        2.the prediction accuracy meets the requirements.
    """
    # model train and predict, case 1 and case 2
    context_model_predict_result = cpkt_transfer_model_context()
    print(f"context_model_predict_result {context_model_predict_result}")

    auto_parallel_model_predict_result = cpkt_transfer_model_auto_parallel()
    print(f"auto_parallel_model_predict_result {auto_parallel_model_predict_result}")

    # functional train and predict, case 3, case4 and case 5
    function_auto_predict_result_dict = cpkt_transfer_func_auto_parallel()
    function_load_distributed_checkpoint = function_auto_predict_result_dict.get("load_distributed_checkpoint")
    function_transform_checkpoints = function_auto_predict_result_dict.get("transform_checkpoints")
    function_load_segmented_checkpoints = function_auto_predict_result_dict.get("load_segmented_checkpoints")
    print("function_predict_result_load_distributed_checkpoint", function_load_distributed_checkpoint)
    print("function_predict_result_transform_checkpoints", function_transform_checkpoints)
    print("function_predict_result_load_segmented_checkpoints", function_load_segmented_checkpoints)

    # accuracy
    compare_params(function_load_distributed_checkpoint, context_model_predict_result)
    compare_params(function_transform_checkpoints, context_model_predict_result)
    compare_params(function_load_segmented_checkpoints, context_model_predict_result)
    compare_params(auto_parallel_model_predict_result, context_model_predict_result)
