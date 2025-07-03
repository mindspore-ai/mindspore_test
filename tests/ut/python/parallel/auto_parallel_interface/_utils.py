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

import os
import glob
import re
import shutil
import subprocess

from mindspore.parallel.auto_parallel import AutoParallel
from hccl_test.manage.api import Hccl


# init hccl
def init_hccl(global_rank, device_num):
    hccl = Hccl()
    hccl.rank_id = global_rank
    hccl.rank_size = device_num


# set auto_parallel mode
def set_parallel_mode(obj, parallel_config=None):
    if parallel_config is None:
        return obj
    parallel_mode = parallel_config.get("parallel_mode", "semi_auto")
    net = AutoParallel(obj, parallel_mode)
    if parallel_config.get("dataset_strategy", None) is not None:
        net.dataset_strategy(parallel_config["dataset_strategy"])
    if parallel_config.get("comm_fusion", None) is not None:
        net.comm_fusion(parallel_config["comm_fusion"])
    if parallel_config.get("dump_local_norm", None) is True:
        net.print_local_norm()
    if parallel_config.get("enable_parallel_optimizer", None) is True:
        net.hsdp()
    if parallel_config.get("parallel_optimizer_config", None) is not None:
        opt_config = parallel_config["parallel_optimizer_config"]
        shard_size = opt_config.get("optimizer_weight_shard_size", -1)
        threshold = opt_config.get("parallel_optimizer_threshold", 64)
        optimizer_level = opt_config.get("optimizer_level", "level1")
        net.hsdp(shard_size, threshold, optimizer_level)
    if parallel_config.get("force_fp32_communication", None) is True:
        net.enable_fp32_communication()
    if parallel_config.get("gradients_mean", None) is True:
        net.enable_gradients_mean()
    if parallel_config.get("gradient_fp32_sync", None) is False:
        net.disable_gradient_fp32_sync()
    if parallel_config.get("group_ckpt_save_file", None) is not None:
        net.set_group_ckpt_save_file(parallel_config["group_ckpt_save_file"])
    if parallel_config.get("loss_repeated_mean", None) is False:
        net.disable_loss_repeated_mean()
    if parallel_config.get("save_strategy_file_path", None) is not None:
        net.save_param_strategy_file(parallel_config["save_strategy_file_path"])
    if parallel_config.get("pipeline_config", None) is not None:
        pipeline_config = parallel_config["pipeline_config"]
        stages = pipeline_config.get("stages", 1)
        output_broadcast = pipeline_config.get("output_broadcast", False)
        interleave = pipeline_config.get("interleave", False)
        scheduler = pipeline_config.get("scheduler", "1f1b")
        net.pipeline(stages, output_broadcast, interleave, scheduler)
    return net


def save_ir_graphs(file_name):
    exec_path = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(exec_path, file_name)
    os.environ['MS_DEV_SAVE_GRAPHS'] = "2"
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = file_path
    return file_path


# Delete folders with specific keyword names under base_dir
def remove_files(keyword, base_dir):
    folder_paths = glob.glob(os.path.join(base_dir, '*'))
    pattern = re.compile(rf'{keyword}')
    file_paths = [path for path in folder_paths if pattern.search(os.path.basename(path)) and os.path.isdir(path)]
    if file_paths:
        for folder in file_paths:
            shutil.rmtree(folder, ignore_errors=True)


# Get the file path of the .ir with the largest file size whose file name contains keyword
def find_ir_file_path(graph_path, file_name_keyword):
    largest_size = 0
    ir_graph_name = None
    root_of_largest_file = None  # Store the root of the largest file

    for root, _, files in os.walk(graph_path):
        for file in files:
            if file.endswith('.ir') and file_name_keyword in file:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)

                if file_size > largest_size:
                    largest_size = file_size
                    ir_graph_name = file
                    root_of_largest_file = root  # Update root with the current directory

    if ir_graph_name is None:
        raise ValueError(f"No IR file found with the keyword '{file_name_keyword}' in {graph_path}")

    # Ensure that root_of_largest_file is defined before creating the file_path
    file_path = os.path.join(root_of_largest_file, ir_graph_name)
    print(f"file_path is {file_path}")
    return file_path


# check the number of attrs of nodes in the graph file
def check_node_attrs_pair(file_path, check_pairs):
    # check_pairs = {'node_name': {'key_word1': '1', 'key_word2': '2'}}
    if check_pairs is None:
        raise ValueError("check_pairs is None")
    for node_name, sub_dict in check_pairs.items():
        for key_word, expected_value in sub_dict.items():
            try:
                grep_command = f"grep '{node_name}' {file_path} | grep '{key_word}'"
                grep_output = subprocess.check_output(grep_command, shell=True)
            except subprocess.CalledProcessError as e:
                print(f"Fail to find node because {e}")
                raise ValueError(f"Failed to find {node_name} in {file_path}")
            split_grep_output = str(grep_output, 'utf-8').strip().split("\n")
            appear_count = len(split_grep_output)
            assert appear_count == expected_value, (f"The pattern {sub_dict} appears {appear_count}, "
                                                    f"expect {expected_value}")


# Check the relationship between the input node and the target node by reversely searching.
def check_node_dependency_backward_search(file_path, backward_lines, dependency_list):
    # dependency_list = [start_unique_node, use_node_idx, use_node_idx, ..., end_node_name]
    start_node = dependency_list.pop(0)
    # Matching lines and their preceding backward_lines lines will be output.
    matched_start_lines = subprocess.check_output(
        [f"grep '{start_node}' {file_path} -B {backward_lines}"],
        shell=True)
    if not matched_start_lines:
        raise ValueError(f"Failed to find {start_node} in {file_path}")
    split_matched_start_lines = str(matched_start_lines, 'utf-8').strip().split("\n")

    find_node_mark = re.findall(r"%\d+", split_matched_start_lines[-1])[0]
    for line in reversed(split_matched_start_lines):
        # find all %n mark in current node, remove the first one (self)
        current_node_mark = re.findall(r"%\d+", line)
        if not current_node_mark:
            continue
        # match the first %n mark in current node, continue to search its use_node
        if find_node_mark == current_node_mark[0]:
            all_use_node_mark = current_node_mark[1:]
            use_idx = dependency_list.pop(0)
            # The last value of dependency_list is the end node name
            if not dependency_list:
                # if the end node name is not in the last line, return False
                assert use_idx in line, f"Failed to find {use_idx} in {line}"
                break
            else:
                # find use node mark
                if use_idx >= len(all_use_node_mark):
                    raise ValueError(f"{use_idx} is out of range, all_use_node_mark is {all_use_node_mark}")
                find_node_mark = all_use_node_mark[use_idx]
        else:
            continue
    if dependency_list:
        raise ValueError(f"Failed to find all dependency nodes in {file_path}")


def check_node_pairs_num(file_path, check_pairs):
    if check_pairs is None:
        raise ValueError("check_pairs is None")
    for node_name, value in check_pairs.items():
        grep_command = ["grep -r '%s' %s | wc -l" % (node_name, file_path)]
        grep_output = subprocess.check_output(grep_command, shell=True)
        if not grep_output:
            raise ValueError(f"Failed to find {node_name} in {file_path}")
        appear_count = str(grep_output, 'utf-8').strip()
        assert appear_count == value, f"The pattern {node_name} appears {appear_count}, expect {value}"
