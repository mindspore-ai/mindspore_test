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
import numpy as np

from mindspore import Tensor
from mindspore.communication.management import get_rank, get_group_size
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
    return net


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


class FakeDataInitMode:
    RandomInit = 0
    OnesInit = 1
    UniqueInit = 2
    ZerosInit = 3

class FakeData3:
    """A fake dataset that returns randomly generated images and returns them as PIL images
       image data type is np.float32 in default
       label data type is np.int64 in default
       label data is onehot in default
       weight_data type is np.float32 in default

    Args:
        size (int, optional): size of the dataset. Default: 1024 images
        batch_size (int, optional): how many samples per batch to load. Default: 32 images
        image_size(tuple, optional): size if the returned images. Default: (3, 224, 224)
        num_classes(int, optional): number of classes in the dataset. Default: 10
        random_offset (int): offsets the index-based random seed used to
            generate each image. Default: 0

    """

    def __init__(self, size=1024, batch_size=32, image_size=(3, 224, 224),
                 num_classes=10, random_offset=0, use_parallel=False, rol=0.0001,
                 fakedata_mode=FakeDataInitMode.RandomInit, image_dtype=np.float16,
                 label_dtype=np.int64, weight_dtype=np.float16):
        self.size = size
        self.rank_batch_size = batch_size
        self.total_batch_size = self.rank_batch_size
        self.random_offset = random_offset
        self.image_size = image_size
        self.weight_size = num_classes  # batch_c
        self.num_classes = num_classes
        self.rol = rol
        self.rank_size = 1
        self.rank_id = 0
        self.batch_index = 0
        self.image_data_type = image_dtype
        self.label_data_type = label_dtype
        self.weight_data_type = weight_dtype  # weight
        self.is_onehot = True
        self.fakedata_mode = fakedata_mode

        if use_parallel is True:
            self.rank_size = get_group_size()
            self.rank_id = get_rank()
            self.weight_size = num_classes // self.rank_size
        self.total_batch_size = self.rank_batch_size * self.rank_size

        assert self.size % self.total_batch_size == 0

        self.total_batch_data_size = (self.rank_size, self.rank_batch_size) + image_size
        self.total_num_class_data_size = (self.rank_size, self.weight_size)

    def get_dataset_size(self):
        return int(self.size / self.total_batch_size)

    def get_repeat_count(self):
        return 1

    def set_image_data_type(self, data_type):
        self.image_data_type = data_type

    def set_label_data_type(self, data_type):
        self.label_data_type = data_type

    def set_weight_data_type(self, data_type):
        self.weight_data_type = data_type

    def set_label_onehot(self, is_onehot=True):
        self.is_onehot = is_onehot

    def create_tuple_iterator(self, num_epochs=-1, do_copy=False):
        return self

    def __getitem__(self, batch_index):
        if batch_index * self.total_batch_size >= len(self):
            raise IndexError("{} index out of range".format(self.__class__.__name__))
        rng_state = np.random.get_state()
        np.random.seed(batch_index + self.random_offset)
        if self.fakedata_mode == FakeDataInitMode.OnesInit:
            img = np.ones(self.total_batch_data_size)
            weight = np.ones(self.total_num_class_data_size)
        elif self.fakedata_mode == FakeDataInitMode.ZerosInit:
            img = np.zeros(self.total_batch_data_size)
            weight = np.zeros(self.total_num_class_data_size)
        elif self.fakedata_mode == FakeDataInitMode.UniqueInit:
            total_size = 1
            for i in self.total_batch_data_size:
                total_size = total_size * i
            img = np.reshape(np.arange(total_size) * self.rol, self.total_batch_data_size)
            weight = np.reshape(np.arange(total_size) * self.rol, self.total_num_class_data_size)
        else:
            img = np.random.randn(*self.total_batch_data_size)
            weight = np.random.randn(*self.total_num_class_data_size)
        target = np.random.randint(0, self.num_classes, size=(self.rank_size, self.rank_batch_size))

        # cur_rank data
        np.random.set_state(rng_state)
        img = img[self.rank_id] # [self.rank_batch_size, self.image_size]
        weight = weight[self.rank_id]
        target = target[self.rank_id]  # [self.rank_batch_size, ], fill with 0~self.num_classes

        # data_type
        img_ret = img.astype(self.image_data_type)
        weight_ret = weight.astype(self.weight_data_type)
        target_ret = target.astype(self.label_data_type)

        # one-hot target
        if self.is_onehot:
            target_onehot = np.zeros(shape=(self.rank_batch_size, self.num_classes))  # [rank_batch_size, num_classes]
            target_onehot[np.arange(self.rank_batch_size), target] = 1  # [rank_batch_size, num_classes]
            target_ret = target_onehot.astype(self.label_data_type)
        return Tensor(img_ret), Tensor(target_ret), Tensor(weight_ret)

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
