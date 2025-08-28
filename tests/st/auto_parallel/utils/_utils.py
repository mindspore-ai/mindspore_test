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

import os
import shutil
import numpy as np

from mindspore import log as logger
from mindspore.train import get_ckpt_path_with_strategy
from mindspore.parallel.auto_parallel import AutoParallel


# set auto_parallel mode by auto_parallel interface
def set_parallel_mode(obj, parallel_config=None):
    if parallel_config is None:
        return obj
    parallel_mode = parallel_config.get("parallel_mode", "semi_auto")
    net = AutoParallel(obj, parallel_mode)
    if parallel_config.get("dataset_strategy", None) is not None:
        net.dataset_strategy(parallel_config["dataset_strategy"])
    if parallel_config.get("pipeline_config", None) is not None:
        pipeline_config = parallel_config["pipeline_config"]
        stages = pipeline_config.get("stages", 1)
        output_broadcast = pipeline_config.get("output_broadcast", False)
        interleave = pipeline_config.get("interleave", False)
        scheduler = pipeline_config.get("scheduler", "1f1b")
        net.pipeline(stages, output_broadcast, interleave, scheduler)
    if parallel_config.get("save_strategy_file", None) is not None:
        net.save_param_strategy_file(parallel_config["save_strategy_file"])
    if parallel_config.get("load_strategy_file", None) is not None:
        net.load_param_strategy_file(parallel_config["load_strategy_file"])
    if parallel_config.get("enable_parallel_optimizer", None) is True:
        net.hsdp()
    if parallel_config.get("parallel_optimizer_config", None) is not None:
        opt_config = parallel_config["parallel_optimizer_config"]
        shard_size = opt_config.get("optimizer_weight_shard_size", -1)
        threshold = opt_config.get("parallel_optimizer_threshold", 64)
        optimizer_level = opt_config.get("optimizer_level", "level1")
        net.hsdp(shard_size, threshold, optimizer_level)
    if parallel_config.get("ascend_config", None) is not None:
        net.transformer_opt(parallel_config["ascend_config"])
    return net


# clean ckpt files
def clean_all_ckpt_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ckpt') or file_name.endswith('.meta'):
                try:
                    os.remove(os.path.join(folder_path, file_name))
                except FileNotFoundError as e:
                    logger.warning("[{}] remove ckpt file error.".format(e))


# clean file in directory_path
def clear_files_in_directory(directory_path):
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        logger.info(f"The path {directory_path} does not exist or is not a directory.")
        return

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


# find ckpt
def find_newest_ckpt_file(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith('.ckpt'),
                            os.listdir(folder_path)))
    return max(ckpt_files, key=os.path.getctime)


def find_newest_ckpt_file_by_name(folder_path):
    ckpt_files = map(lambda f: os.path.join(folder_path, f),
                     filter(lambda f: f.endswith('.ckpt'),
                            os.listdir(folder_path)))
    return max(list(ckpt_files))


# find the available checkpoint file and return the paths.
def parallel_mode_get_ckpt_path_with_strategy(strategy_file=None, cpkt_path=None):
    ckpt_file = find_newest_ckpt_file(cpkt_path)
    ckpt_file_new = get_ckpt_path_with_strategy(ckpt_file, strategy_file)
    logger.info(f"Find checkpoint file: {ckpt_file_new}")


def save_ir_graphs(file_path):
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = file_path


# compare accuracy
def compare_params(ex_params, actual_params):
    assert np.allclose(ex_params.asnumpy(), actual_params.asnumpy(), atol=1e-3, rtol=1e-3)


def compare_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_expected, data_me, rtol, atol, equal_nan=equal_nan):
        _count_unequal_element(data_expected, data_me, rtol, atol)
    else:
        assert np.array(data_expected).shape == np.array(data_me).shape


def _count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_me) * rtol)
    nan_diff = np.not_equal(np.isnan(data_expected), np.isnan(data_me))
    inf_diff = np.not_equal(np.isinf(data_expected), np.isinf(data_me))
    neginf_diff = np.not_equal(np.isneginf(data_expected), np.isneginf(data_me))
    greater = greater + nan_diff + inf_diff + neginf_diff
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
            format(data_expected[greater], data_me[greater], error[greater])
