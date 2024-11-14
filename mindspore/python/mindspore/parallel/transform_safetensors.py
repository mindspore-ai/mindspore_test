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
"""Transform distributed safetensors"""
from __future__ import absolute_import

import os
import time
import glob
import re
import math
import json
from collections import defaultdict

import multiprocessing as mp
import numpy as np
import mindspore as ms
from mindspore.parallel._parallel_serialization import _get_device_num_from_strategy, _make_dir, \
    _extract_layout_map, _extract_src_dst_layout_map, _parameter_not_in_local_stage, _extract_pipeline_stage_num, \
    _insert_opt_shard_reshape, _extract_src_dst_layout_map_by_src
from mindspore.parallel._tensor import _get_tensor_strategy, _construct_from_to_tensor_layout, \
    _get_needed_rank_transform_operator_map_by_layouts, \
    _generate_transform_operator_stack, _apply_tensor_transform_operators, _construct_tensor_layout_for_opt_shard, \
    _extract_layout_item, _load_tensor_shape, _apply_operator
from mindspore.parallel._parallel_serialization import _build_searched_strategy, _load_protobuf_strategy, \
    _convert_to_list

from safetensors.numpy import save_file, load_file
from safetensors import safe_open


def _load_and_transform(path, name_map, load_func, transform_func):
    if load_func is not None:
        param_dict = load_func(path)
    else:
        param_dict = path
    transform_dict = {}
    for k, v in param_dict.items():
        new_name = name_map.get(k, k) if name_map is not None else k
        transform_dict[new_name] = transform_func(v, new_name)
    return transform_dict


def _transform_tensor_to_numpy(path, name_map=None):
    return _load_and_transform(path, name_map, ms.load_checkpoint, lambda v, new_name: v.asnumpy())


def _transform_numpy_to_tensor(path, name_map=None):
    return _load_and_transform(path, name_map, load_file, lambda v, new_name: ms.Parameter(v, name=new_name))


def _process_file(file_info):
    cur_ckpt_path, name_map, save_path, file = file_info
    param_dict_numpy = _transform_tensor_to_numpy(cur_ckpt_path, name_map)
    safetensors_filename = file.replace(".ckpt", ".safetensors")
    dst_file = os.path.join(save_path, safetensors_filename)
    save_file(param_dict_numpy, dst_file)


def _process_file_safetensors(file_info):
    cur_safe_path, name_map, save_path, file = file_info
    param_dict_tensor = _transform_numpy_to_tensor(cur_safe_path, name_map)
    ckpt_filename = file.replace(".safetensors", ".ckpt")
    dst_file = os.path.join(save_path, ckpt_filename)
    ms.save_checkpoint(param_dict_tensor, dst_file)


def _gather_tasks(file_path, save_path, file_name_regex, name_map):
    """gather transform rank together"""
    tasks = []
    for root, dirs, _ in os.walk(file_path):
        if root != file_path:
            continue

        rank_dirs = [d for d in dirs if d.startswith('rank')]
        if not rank_dirs:
            raise ValueError(
                f"For 'ckpt_to_safetensors', no directories starting with 'rank' found in {file_path}")

        for rank_dir in rank_dirs:
            rank_dir_path = os.path.join(root, rank_dir)
            dst_root = os.path.join(save_path,
                                    os.path.relpath(rank_dir_path, file_path)) if save_path else rank_dir_path
            os.makedirs(dst_root, exist_ok=True)
            tasks.extend(
                (os.path.join(rank_dir_path, file), name_map, dst_root, file)
                for file in os.listdir(rank_dir_path)
                if file.endswith(".ckpt") and (file_name_regex is None or re.findall(file_name_regex, file))
            )
    return tasks


def _progress_bar(iterable, total=None):
    """
    Decorate an iterable object, returning an iterator which acts exactly
    like the original iterable, but prints a dynamically updating
    progressbar every time a value is requested.
    """
    if total is None:
        total = len(iterable)

    start_time = time.time()

    def print_progress_bar(iteration):
        percent = f"{100 * (iteration / float(total)):.1f}"
        bar_length = 40
        filled_length = int(bar_length * iteration // total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / iteration * total
        remaining_time = estimated_total_time - elapsed_time

        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        remaining_time_str = time.strftime("%H:%M:%S", time.gmtime(remaining_time))

        print(f'\r{percent}%|{bar}|[{elapsed_time_str}<{remaining_time_str}]', end='')
        if iteration == total:
            print()

    for i, item in enumerate(iterable, start=1):
        yield item
        print_progress_bar(i)


def ckpt_to_safetensors(file_path, save_path=None, name_map=None, file_name_regex=None, processes_num=1):
    """
    Converts MindSpore checkpoint files into safetensors format and saves them to `save_path`.
    Safetensors is a reliable and portable machine learning model storage format introduced by Huggingface,
    used for securely storing Tensors with fast speed (zero copy).

    Note:
        The number of multiprocess settings is related to the size of the host, and it is not recommended to set it
        too large, otherwise it may cause freezing.
        The safetensors format does not support the enc verification function. If ckpt is enabled to save enc
        verification, an error will be generated when performing the conversion.
        The safetensors format currently does not support crc verification function. If ckpt contains crc verification
        information, the crc verification information will be lost after conversion to safetensors.

    Args:
        file_path (str): Path to the directory containing checkpoint files or a single checkpoint file (.ckpt).
        save_path (str, optional): Directory path where safetensors files will be saved. Defaults: ``None``.
        name_map (dict, optional): Dictionary mapping original parameter names to new names. Defaults: ``None``.
        file_name_regex (str, optional): Regular expression used to match the file that needs to be converted.
                                   Defaults: ``None``.
        processes_num (int, optional): Number of processes to use for parallel processing. Defaults: 1.
    Raises:
        ValueError: If the input path is invalid or the save_path is not a directory,
                    or the file_path does not end with '.ckpt'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> ms.ckpt_to_safetensors("./ckpt_save_path")
        >>> ms.ckpt_to_safetensors("./ckpt_save_path/rank0/checkpoint_0.ckpt")
        >>> ms.ckpt_to_safetensors(file_path="./ckpt_save_path/rank0/checkpoint_0.ckpt", save_path="./new_path/")
        >>> namemap = {"lin.weight":"new_name"}
        >>> ms.ckpt_to_safetensors("./ckpt_save_path/rank0/checkpoint_0.ckpt", "./new_path/", namemap)
    """
    is_dir = os.path.isdir(file_path)
    is_file = os.path.isfile(file_path)
    if not is_dir and not is_file:
        raise ValueError(f"For 'ckpt_to_safetensors', the input path must be a valid path or file, but got {file_path}")
    if save_path and os.path.splitext(save_path)[1]:
        raise ValueError(f"For 'ckpt_to_safetensors', the save_path must be a directory, but got '{save_path}'")
    if name_map is not None and not isinstance(name_map, dict):
        raise ValueError(
            f"For 'ckpt_to_safetensors', the type of 'name_map' must be a directory, but got '{type(name_map)}'")

    if is_dir:
        tasks = _gather_tasks(file_path, save_path, file_name_regex, name_map)
        with mp.Pool(processes=processes_num) as pool:
            list(_progress_bar(pool.imap(_process_file, tasks), total=len(tasks)))
    elif is_file:
        if not file_path.endswith(".ckpt"):
            raise ValueError(f"For 'ckpt_to_safetensors', the input file must be a .ckpt file, but got {file_path}")
        if file_name_regex is not None and not re.findall(file_name_regex, file_path):
            raise ValueError(f"For 'ckpt_to_safetensors', the input file does not match the regular expression.")
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        param_dict_numpy = _transform_tensor_to_numpy(file_path, name_map)
        safetensors_filename = os.path.basename(file_path).replace(".ckpt", ".safetensors")
        dst_file = os.path.join(save_path if save_path else os.path.dirname(file_path), safetensors_filename)
        save_file(param_dict_numpy, dst_file)


def _gather_safetensors_tasks(file_path, save_path, file_name_regex, name_map):
    """gather transform rank together"""
    tasks = []
    for root, dirs, _ in os.walk(file_path):
        if root != file_path:
            continue

        rank_dirs = [d for d in dirs if d.startswith('rank')]
        if not rank_dirs:
            raise ValueError(
                f"For 'safetensors_to_ckpt', no directories starting with 'rank' found in {file_path}")

        for rank_dir in rank_dirs:
            rank_dir_path = os.path.join(root, rank_dir)
            dst_root = os.path.join(save_path,
                                    os.path.relpath(rank_dir_path, file_path)) if save_path else rank_dir_path
            os.makedirs(dst_root, exist_ok=True)
            tasks.extend(
                (os.path.join(rank_dir_path, file), name_map, dst_root, file)
                for file in os.listdir(rank_dir_path)
                if file.endswith(".safetensors") and (file_name_regex is None or re.findall(file_name_regex, file))
            )
    return tasks


def safetensors_to_ckpt(file_path, save_path=None, name_map=None, file_name_regex=None, processes_num=1):
    """
    Converts safetensors files into MindSpore checkpoint format and saves them to `save_path`.
    Safetensors is a reliable and portable machine learning model storage format introduced by Huggingface,
    used for securely storing Tensors with fast speed (zero copy).

    Note:
        The number of multiprocess settings is related to the size of the host, and it is not recommended to set it
        too large, otherwise it may cause freezing.

    Args:
        file_path (str): Path to the directory containing safetensors files or a single safetensors file (.safetensors).
        save_path (str, optional): Directory path where checkpoint files will be saved. Defaults: ``None``.
        name_map (dict, optional): Dictionary mapping original parameter names to new names. Defaults: ``None``.
        file_name_regex (str, optional): Regular expression used to match the file that needs to be converted.
                                   Defaults: ``None``.
        processes_num (int, optional): Number of processes to use for parallel processing. Defaults: 1.

    Raises:
        ValueError: If the input path is invalid, the save_path is not a directory,
                    or the file_path does not end with '.safetensors'.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> ms.safetensors_to_ckpt("./safetensors_save_path")
        >>> ms.safetensors_to_ckpt("./safetensors_save_path/rank0/checkpoint_0.safetensors")
        >>> ms.safetensors_to_ckpt("./safetensors_save_path/rank0/checkpoint_0.safetensors", "./new_path/")
        >>> namemap = {"lin.weight":"new_name"}
        >>> ms.safetensors_to_ckpt("./safetensors_save_path/rank0/checkpoint_0.safetensors", "./new_path/", namemap)
    """
    is_dir = os.path.isdir(file_path)
    is_file = os.path.isfile(file_path)
    if not is_dir and not is_file:
        raise ValueError(f"For 'safetensors_to_ckpt', the input path must be a valid path or file, but got {file_path}")
    if save_path and os.path.splitext(save_path)[1]:
        raise ValueError(f"For 'safetensors_to_ckpt', the save_path must be a directory, but got '{save_path}'")
    if name_map is not None and not isinstance(name_map, dict):
        raise ValueError(
            f"For 'safetensors_to_ckpt', the type of 'name_map' must be a directory, but got '{type(name_map)}'")

    if is_dir:
        tasks = _gather_safetensors_tasks(file_path, save_path, file_name_regex, name_map)
        with mp.Pool(processes=processes_num) as pool:
            list(_progress_bar(pool.imap(_process_file_safetensors, tasks), total=len(tasks)))
    elif is_file:
        if not file_path.endswith(".safetensors"):
            raise ValueError(
                f"For 'safetensors_to_ckpt', the input file must be a .safetensors file, but got {file_path}")
        if file_name_regex is not None and not re.findall(file_name_regex, file_path):
            raise ValueError(f"For 'safetensors_to_ckpt', the input file does not match the regular expression.")
        if save_path and not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        param_dict_tensor = _transform_numpy_to_tensor(file_path, name_map)
        ckpt_filename = os.path.basename(file_path).replace(".safetensors", ".ckpt")
        dst_file = os.path.join(save_path if save_path else os.path.dirname(file_path), ckpt_filename)
        ms.save_checkpoint(param_dict_tensor, dst_file)


def _check_transform_safetensors(src_safetensors_dir, ckpt_prefix, src_strategy_file, dst_strategy_file):
    """check _transform_safetensors input"""
    if not isinstance(ckpt_prefix, str):
        raise TypeError("The ckpt_prefix should be a str.")
    if src_strategy_file and os.path.dirname(src_strategy_file) and not os.path.exists(
            os.path.dirname(src_strategy_file)):
        raise ValueError("The director of src_strategy_file: {} is not exists.".
                         format(os.path.dirname(src_strategy_file)))
    if dst_strategy_file and os.path.dirname(dst_strategy_file) and not os.path.exists(
            os.path.dirname(dst_strategy_file)):
        raise ValueError("The director of dst_strategy_file: {} is not exists.".
                         format(os.path.dirname(dst_strategy_file)))


def _check_output_format(output_format):
    if output_format not in ["safetensors", "ckpt"]:
        raise ValueError(f"For 'transform_safetensors', the output_format must be "
                         f"'safetensors' or 'ckpt', but got {output_format}.")


def _split_protobuf_strategy(merged_strategy_file):
    """split src_strategy_file by pp"""
    dst_parallel_strategy_map = _load_protobuf_strategy(merged_strategy_file)
    if not dst_parallel_strategy_map.parallel_strategy_item or not dst_parallel_strategy_map.parallel_layout_item:
        raise ValueError(f"The merged strategy file {merged_strategy_file} is empty")

    src_dict = {}
    for layout_item in dst_parallel_strategy_map.parallel_layout_item:
        stage, _ = layout_item.param_name.split('-', 1)
        stage = int(stage)
        if stage not in src_dict:
            src_dict[stage] = {}
        parameter_name = layout_item.param_name
        layout = layout_item.parallel_layouts
        src_dict[stage][parameter_name] = layout
    return src_dict


def _transform_safetensors(src_safetensors_dir, dst_safetensors_dir, ckpt_prefix, src_strategy_file=None,
                           dst_strategy_file=None, process_num=1, output_format="safetensors"):
    """Transform distributed safetensors from source sharding strategy to destination sharding strategy for a rank."""
    _check_transform_safetensors(src_safetensors_dir, ckpt_prefix, src_strategy_file, dst_strategy_file)
    _check_output_format(output_format)
    _make_dir(dst_safetensors_dir, "path")
    all_safetensor_files_map = _collect_safetensor_files(src_safetensors_dir)

    dst_strategy_dict = _build_searched_strategy(dst_strategy_file)
    pipeline_stage_num = _extract_pipeline_stage_num(src_strategy_file)
    dst_stage_num = _extract_pipeline_stage_num(dst_strategy_file)

    if pipeline_stage_num > 1 and dst_stage_num == 1:
        stage_dict = _split_protobuf_strategy(src_strategy_file)

        processes = []
        manager = mp.Manager()
        _transform_param_list = manager.list()
        for _, src_strategy_dict in stage_dict.items():
            p = mp.Process(target=_transform_stage_safetensors,
                           args=(src_strategy_dict, dst_strategy_dict, ckpt_prefix,
                                 dst_safetensors_dir, output_format, all_safetensor_files_map, process_num,
                                 _transform_param_list))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        _save_final_safetensors(_transform_param_list, output_format)
    else:
        src_strategy_dict = _build_searched_strategy(src_strategy_file)
        _transform_stage_safetensors(src_strategy_dict, dst_strategy_dict, ckpt_prefix,
                                     dst_safetensors_dir, output_format, all_safetensor_files_map, process_num,
                                     _transform_param_list=None)


def _transform_stage_safetensors(src_strategy_dict, dst_strategy_dict, ckpt_prefix,
                                 dst_safetensors_dir, output_format, all_safetensor_files_map, process_num,
                                 _transform_param_list):
    """Transform distributed safetensors by stage"""
    src_stage_device_num = _get_device_num_from_strategy(src_strategy_dict)
    dst_stage_device_num = _get_device_num_from_strategy(dst_strategy_dict)

    origin_src_strategy_list = _extract_layout_map(src_strategy_dict)
    origin_dst_strategy_list = _extract_layout_map(dst_strategy_dict)

    needed_rank_list_map = _find_needed_ranks(src_strategy_dict, dst_strategy_dict)
    for needed_rank_list, rank in needed_rank_list_map.items():
        for needed_rank in needed_rank_list.split("-"):
            if int(needed_rank) not in all_safetensor_files_map:
                raise ValueError("The safetensor file of rank{} is needed for converting rank{}'s safetensor, "
                                 "but it is missing.".format(needed_rank, rank))
    dst_stage_num = _extract_pipeline_stage_num(dst_strategy_dict)
    if not (len(needed_rank_list_map) == 1 and dst_stage_num > 1) and process_num > len(needed_rank_list_map):
        ms.log.warning("The value of process_num cannot be greater than that of needed_rank_list_map.")
        process_num = len(needed_rank_list_map)
    _transform_safetensors_with_parallel(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                         dst_stage_device_num, src_strategy_dict, dst_strategy_dict,
                                         origin_src_strategy_list, origin_dst_strategy_list, ckpt_prefix,
                                         dst_safetensors_dir, process_num, output_format,
                                         _transform_param_list)


def _distribute_files_by_size(all_safetensor_files_map, needed_rank_list_map, process_num):
    """
    Distributes files across multiple processes based on file size to balance the processing load.
    """
    if process_num == 1:
        return [needed_rank_list_map]
    # Calculate the size of each file.
    # if src==1, dst pp>1, split for pp number.
    if len(needed_rank_list_map) == 1:
        src_rank = next(iter(needed_rank_list_map.keys()))
        dst_list = next(iter(needed_rank_list_map.values()))
        size = len(dst_list) // process_num
        split_list = [dst_list[i:i + size] for i in range(0, len(dst_list), size)]
        part_list_dict = [dict() for _ in range(process_num)]
        for index in range(process_num):
            part_list_dict[index][src_rank] = split_list[index]
        return part_list_dict

    rank_size = dict()
    for rank_id, file_name in all_safetensor_files_map.items():
        tmp_size = os.path.getsize(file_name) / 1024 / 1024
        rank_size[rank_id] = tmp_size
    # Obtain the rank and size required by all parts.
    part_total = []
    for index, (k, v) in enumerate(needed_rank_list_map.items()):
        tmp_part = []
        key_ele = k.split("-")
        tmp_size = 0
        for ele in key_ele:
            tmp_size += rank_size[int(ele)]
        tmp_part.append(index)
        tmp_part.append(tmp_size)
        part_total.append(tmp_part)
    # Sort each part by size.
    part_total = sorted(part_total, key=lambda x: x[1], reverse=True)
    part_list = [[] for _ in range(process_num)]
    part_size = [[] for _ in range(process_num)]
    for [index, size] in part_total:
        min_sum = float('inf')
        min_idx = -1
        for ele in range(process_num):
            if sum(part_size[ele]) < min_sum:
                min_sum = sum(part_size[ele])
                min_idx = ele
        part_list[min_idx].append(index)
        part_size[min_idx].append(size)

    part_list_dict = [dict() for _ in range(process_num)]
    for index, (k, v) in enumerate(needed_rank_list_map.items()):
        for idd, ele in enumerate(part_list):
            if index in ele:
                part_list_dict[idd][k] = v
                break
    return part_list_dict


def _transform_safetensors_with_parallel(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                         dst_stage_device_num, src_strategy_dict, dst_strategy_dict,
                                         origin_src_strategy_list, origin_dst_strategy_list, ckpt_prefix,
                                         dst_safetensors_dir, process_num, output_format,
                                         _transform_param_list):
    """
    Transforms safetensors files to a specified format using parallel processing.
    """
    # cal param name for every pipeline, save in pipe_param_list.
    pipe_num = _extract_pipeline_stage_num(dst_strategy_dict)
    pipe_param_list = [None for _ in range(max(pipe_num, process_num))]
    if len(needed_rank_list_map) == 1 and pipe_num > 1:
        process_num = pipe_num
        pipe_param_list = [[] for _ in range(pipe_num)]
        layout_map = _convert_to_list(dst_strategy_dict)

        for name, layout in layout_map.items():
            pipe_param_list[layout[6][0]].append(name)

    part_list_dict = _distribute_files_by_size(all_safetensor_files_map, needed_rank_list_map, process_num)
    processes = []
    for i in range(process_num):
        p = mp.Process(target=_transform_safetensors_single, args=(
            part_list_dict[i], all_safetensor_files_map, src_stage_device_num, dst_stage_device_num,
            src_strategy_dict, dst_strategy_dict, origin_src_strategy_list, origin_dst_strategy_list,
            ckpt_prefix, dst_safetensors_dir, output_format, _transform_param_list, pipe_param_list[i]))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def _count_redundancy_list(rank_num, param_name, redundancy_dict, device_num):
    """Obtain the specified redundant group."""
    redundancy_tuple = redundancy_dict.get(param_name)
    for rank_list in redundancy_tuple:
        for rank in rank_list:
            if rank_num % device_num == rank % device_num:
                return set(rank_list)
    return set()


def _find_remove_redundancy_rank_id(pipe_param_list, single_param_dict, file_dict, saftensor_dict, redundancy_dict,
                                    needed_rank, device_num):
    """Find the rank_id under redundant groups."""
    for param_name in pipe_param_list:
        rank_num = int(needed_rank)
        redundancy_ranks = _count_redundancy_list(rank_num, param_name, redundancy_dict, device_num)
        open_file_id = None
        if single_param_dict.get(param_name) is None:
            continue
        for real_rank in single_param_dict[param_name]:
            for redundancy_rank in redundancy_ranks:
                if real_rank % device_num == redundancy_rank % device_num:
                    open_file_id = real_rank
                    break
        if open_file_id is not None:
            output = file_dict[open_file_id].get_tensor(param_name)
            saftensor_dict[param_name] = output
        else:
            raise ValueError(f"For _transform_safetensors_single, {param_name} should be in "
                             f"{redundancy_ranks}, but in {single_param_dict[param_name]}.")


def _transform_safetensors_single(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                  dst_stage_device_num,
                                  src_strategy_dict, dst_strategy_dict, origin_src_strategy_list,
                                  origin_dst_strategy_list,
                                  ckpt_prefix, dst_safetensors_dir, output_format,
                                  _transform_param_list, pipe_param_list=None, file_index=None, unified_flag=False,
                                  src_strategy_file=None):
    """
    Transforms safetensors files to a specified format without using parallel processing.
    """
    if src_strategy_file is not None:
        from mindspore.train._utils import get_parameter_redundancy
        redundancy_dict_tmp = get_parameter_redundancy(src_strategy_file)
        redundancy_dict = {}
        device_num = 0
        for param_name, redundancy in redundancy_dict_tmp.items():
            if device_num == 0:
                device_num = max(max(redundancy)) + 1
            origin_param_name = param_name
            pipeline_stage = 0
            if "-" in param_name:
                pipeline_stage, origin_param_name = param_name.split("-")
                pipeline_stage = int(pipeline_stage)
            redundancy_new = tuple(
                (tuple(x + pipeline_stage * device_num for x in subtuple)) for subtuple in redundancy)
            redundancy_dict[origin_param_name] = redundancy_new
        file_dict = {}
        single_param_dict = {}
        for file_id, _ in all_safetensor_files_map.items():
            f = safe_open(all_safetensor_files_map.get(file_id), framework="np")
            file_dict[file_id] = f
            for param_name in f.keys():
                if param_name not in single_param_dict.keys():
                    single_param_dict[param_name] = {file_id}
                else:
                    single_param_dict[param_name].add(file_id)
    src_strategy_list_keys = _convert_to_list(src_strategy_dict).keys() if src_strategy_dict else []
    dst_strategy_list_keys = _convert_to_list(dst_strategy_dict).keys() if dst_strategy_dict else []
    for needed_rank_list_key, transform_rank_list in needed_rank_list_map.items():
        param_total_dict = defaultdict(dict)
        param_attr_dict = defaultdict(dict)
        needed_rank_list = needed_rank_list_key.split("-")
        for needed_rank in needed_rank_list:
            if pipe_param_list:
                saftensor_dict = dict()
                if src_strategy_file is not None:
                    _find_remove_redundancy_rank_id(pipe_param_list, single_param_dict, file_dict, saftensor_dict,
                                                    redundancy_dict, needed_rank, device_num)
                else:
                    with safe_open(all_safetensor_files_map.get(int(needed_rank)), framework="np") as f:
                        if not unified_flag:
                            all_param_name_set = set(f.keys())
                            src_param_name_set = set(src_strategy_list_keys)
                            dst_param_name_set = set(dst_strategy_list_keys)
                            hyper_param_set = all_param_name_set - (src_param_name_set & dst_param_name_set)
                            pipe_param_list.extend(list(hyper_param_set))
                        for param_name in pipe_param_list:
                            if param_name not in f.keys():
                                # param not in ckpt file, check reason
                                continue
                            output = f.get_tensor(param_name)
                            saftensor_dict[param_name] = output
            else:
                saftensor_dict = load_file(all_safetensor_files_map.get(int(needed_rank)))
            for param_name, param in saftensor_dict.items():
                src_rank = int(needed_rank) % src_stage_device_num
                param_total_dict[param_name][src_rank] = param
                param_attr_dict[param_name][src_rank] = (True, False)

        for transform_rank in transform_rank_list:
            param_total_dict_keys = list(param_total_dict.keys())
            src_strategy_list, dst_strategy_list = _extract_src_dst_layout_map(transform_rank, src_strategy_dict,
                                                                               dst_strategy_dict)
            # cut the parameter not in the pipeline stage.
            for param in list(param_total_dict.keys()):
                if _parameter_not_in_local_stage(param, origin_src_strategy_list, src_strategy_list) \
                        and _parameter_not_in_local_stage(param, origin_dst_strategy_list, dst_strategy_list):
                    param_total_dict_keys.remove(param)

            local_rank_id = transform_rank % dst_stage_device_num
            transform_param_dict = _transform_parallel_safetensor(local_rank_id, param_total_dict,
                                                                  param_attr_dict, src_strategy_list, dst_strategy_list,
                                                                  param_total_dict_keys, src_strategy_file)
            if file_index is not None:
                save_safetensor_file = f"part{file_index}.{output_format}"
                save_safetensor_file_dir = dst_safetensors_dir
            else:
                save_safetensor_file = f"{ckpt_prefix}{transform_rank}.{output_format}"
                save_safetensor_file_dir = os.path.join(dst_safetensors_dir, "rank_{}".format(transform_rank))

            if not os.path.exists(save_safetensor_file_dir):
                _make_dir(save_safetensor_file_dir, "path")
            save_file_name = os.path.join(save_safetensor_file_dir, save_safetensor_file)
            if _transform_param_list is not None:
                _transform_param_list.append({save_file_name: transform_param_dict})
            else:
                if output_format == "safetensors":
                    save_file(transform_param_dict, save_file_name)
                else:
                    transform_param_dict = _load_and_transform(transform_param_dict, None, None,
                                                               transform_func=lambda v, name: ms.Parameter(v,
                                                                                                           name=name))
                    ms.save_checkpoint(transform_param_dict, save_file_name)
            del param_total_dict_keys
        del param_total_dict


def _save_final_safetensors(_transform_param_list, output_format):
    """save file with list"""
    new_transform_dict = {}
    for transform_dict in _transform_param_list:
        for save_file_name, transform_param_dict in transform_dict.items():
            if save_file_name not in new_transform_dict:
                new_transform_dict[save_file_name] = transform_param_dict
            else:
                new_transform_dict[save_file_name].update(transform_param_dict)
    for save_file_name, transform_param_dict in new_transform_dict.items():
        if output_format == "safetensors":
            save_file(transform_param_dict, save_file_name)
        else:
            transform_param_dict = _load_and_transform(transform_param_dict, None, None,
                                                       transform_func=lambda v, name: ms.Parameter(v, name=name))
            ms.save_checkpoint(transform_param_dict, save_file_name)


def transform_safetensors_by_stage(src_safetensors_dir, dst_safetensors_dir, ckpt_prefix,
                                   src_strategy_file,
                                   dst_strategy_file=None):
    """Transform safetensor for stage in src_strategy_file"""
    param_total_dict = defaultdict(dict)
    param_attr_dict = defaultdict(dict)
    param_type_dict = defaultdict(dict)
    src_strategy_list, dst_strategy_list, stage_id = _extract_src_dst_layout_map_by_src(src_strategy_file, \
                                                                                        dst_strategy_file)
    src_stage_device_num = np.prod(src_strategy_list.get(list(src_strategy_list.keys())[0])[0]) if src_strategy_list \
                                                                                                   is not None else 1
    dst_stage_device_num = np.prod(dst_strategy_list.get(list(dst_strategy_list.keys())[0])[0]) if dst_strategy_list \
                                                                                                   is not None else 1
    origin_dst_strategy_list = _extract_layout_map(dst_strategy_file)
    origin_src_strategy_list = _extract_layout_map(src_strategy_file)
    safetensor_files_map = {}
    src_rank_id_start = stage_id * src_stage_device_num
    for local_rank in range(src_stage_device_num):
        rank_id = src_rank_id_start + local_rank
        safetensor_file_name = os.path.join(src_safetensors_dir, "rank_{}".format(rank_id), "*.safetensors")
        rank_ckpts = glob.glob(safetensor_file_name)
        rank_ckpts.sort()
        for safetensor_file in rank_ckpts:
            if not os.path.isfile(safetensor_file):
                continue
            safetensor_files_map[rank_id] = safetensor_file
    for rank, local_file in safetensor_files_map.items():
        if not os.path.exists(local_file):
            raise ValueError("safetensor file {} in rank {} not exits: ".format(local_file, rank))
    for rank, file_name in safetensor_files_map.items():
        saftensor_dict = load_file(file_name)
        for param_name, param in saftensor_dict.items():
            # cut the parameter not in the pipeline stage.
            if _parameter_not_in_local_stage(param_name, origin_src_strategy_list, src_strategy_list) \
                    and _parameter_not_in_local_stage(param_name, origin_dst_strategy_list, dst_strategy_list):
                continue
            src_rank = rank % src_stage_device_num
            param_type_dict[param_name][src_rank] = str(param.data.dtype)
            param_total_dict[param_name][src_rank] = param
            param_attr_dict[param_name][src_rank] = (True, False)
    for local_rank_id in range(dst_stage_device_num):
        transform_param_dict = _transform_parallel_safetensor(local_rank_id, param_total_dict,
                                                              param_attr_dict, src_strategy_list, dst_strategy_list,
                                                              param_type_dict)
        save_safetensor_file = "{}{}_part{}.safetensors".format(ckpt_prefix, local_rank_id, stage_id)
        save_safetensor_file_dir = os.path.join(dst_safetensors_dir, "rank_{}".format(local_rank_id))
        if not os.path.exists(save_safetensor_file_dir):
            _make_dir(save_safetensor_file_dir, "path")
        save_safetensor_file_name = os.path.join(save_safetensor_file_dir, save_safetensor_file)
        save_file(transform_param_dict, save_safetensor_file_name)


def transform_safetensors_by_rank(rank_id, safetensor_files_map, save_safetensor_file_name,
                                  src_strategy_file=None, dst_strategy_file=None):
    """
    Transform distributed checkpoint from source sharding strategy to destination sharding strategy by rank.
    """
    if not isinstance(safetensor_files_map, dict):
        raise TypeError("The safetensor_files_map should be a dict.")
    if not isinstance(rank_id, int):
        raise TypeError("The rank_id should be a int.")
    if not isinstance(save_safetensor_file_name, str):
        raise TypeError("The save_safetensor_file_name should be a str.")
    if not save_safetensor_file_name.endswith(".safetensors"):
        raise ValueError(
            "The save_safetensor_file_name {} should end with .safetensors".format(save_safetensor_file_name))
    if dst_strategy_file and os.path.dirname(dst_strategy_file) and not os.path.exists(
            os.path.dirname(dst_strategy_file)):
        raise ValueError("The director of dst_strategy_file: {} is not exists.".
                         format(os.path.dirname(dst_strategy_file)))
    for rank, local_file in safetensor_files_map.items():
        if not os.path.exists(local_file):
            raise ValueError("safetensor file {} in rank {} not exits: ".format(local_file, rank))
    param_total_dict = defaultdict(dict)
    param_attr_dict = defaultdict(dict)
    param_type_dict = defaultdict(dict)
    src_strategy_list, dst_strategy_list = _extract_src_dst_layout_map(rank_id, src_strategy_file, dst_strategy_file)
    # src rank => local rank inside pipeline stage
    src_stage_device_num = np.prod(src_strategy_list.get(list(src_strategy_list.keys())[0])[0]) if src_strategy_list \
                                                                                                   is not None else 1
    dst_stage_device_num = np.prod(dst_strategy_list.get(list(dst_strategy_list.keys())[0])[0]) if dst_strategy_list \
                                                                                                   is not None else 1
    origin_dst_strategy_list = _extract_layout_map(dst_strategy_file)
    origin_src_strategy_list = _extract_layout_map(src_strategy_file)
    for rank, file_name in safetensor_files_map.items():
        saftensor_dict = load_file(file_name)
        for param_name, param in saftensor_dict.items():
            # cut the parameter not in the pipeline stage.
            if _parameter_not_in_local_stage(param_name, origin_src_strategy_list, src_strategy_list) \
                    and _parameter_not_in_local_stage(param_name, origin_dst_strategy_list, dst_strategy_list):
                continue
            src_rank = rank % src_stage_device_num
            param_type_dict[param_name][src_rank] = str(param.data.dtype)
            # if param.data.dtype == mstype.bfloat16:
            #     param.set_dtype(mstype.float32)
            param_total_dict[param_name][src_rank] = param
            param_attr_dict[param_name][src_rank] = (True, False)
    local_rank_id = rank_id % dst_stage_device_num
    transform_param_dict = _transform_parallel_safetensor(local_rank_id, param_total_dict,
                                                          param_attr_dict, src_strategy_list, dst_strategy_list,
                                                          param_type_dict)
    save_file(transform_param_dict, save_safetensor_file_name)


def _collect_safetensor_files(src_safetensors_dir, format='safetensors', file_suffix=None):
    """
    Collects all safetensors files from the specified directory and its subdirectories.
    """
    if os.path.isfile(src_safetensors_dir) and format == 'safetensors' and src_safetensors_dir.endswith('safetensors'):
        return {0: src_safetensors_dir}
    safetensors_rank_dir_list = os.path.join(src_safetensors_dir, "rank_[0-9]*")
    all_safetensor_files_map = {}
    for safetensor_dir in glob.glob(safetensors_rank_dir_list):
        if not os.path.isdir(safetensor_dir):
            ms.log.warning("{} is not a directory.".format(safetensor_dir))
            continue
        rank_id_str = safetensor_dir.split('rank_')[-1]
        if not rank_id_str.isdigit():
            ms.log.warning("{} is not a expected directory, the directory should end with rank_0/rank_1.....".
                           format(safetensor_dir))
            continue
        rank_id = int(rank_id_str)
        if file_suffix is None:
            safetensor_file_name = os.path.join(safetensor_dir, f"*.{format}")
        else:
            safetensor_file_name = os.path.join(safetensor_dir, f"*{file_suffix}.{format}")
        rank_ckpts = glob.glob(safetensor_file_name)
        rank_ckpts.sort()
        for safetensor_file in rank_ckpts:
            if not os.path.isfile(safetensor_file):
                ms.log.warning("{} is not a safetensor file.".format(safetensor_file))
                continue
            all_safetensor_files_map[rank_id] = safetensor_file
    return all_safetensor_files_map


def _find_needed_ranks(src_strategy_dict, dst_strategy_dict):
    """
    Identifies the ranks needed for transformation based on source and destination strategies.
    """
    needed_rank_list_map = defaultdict(list)
    dst_stage_device_num = _get_device_num_from_strategy(dst_strategy_dict)
    dst_stage_num = _extract_pipeline_stage_num(dst_strategy_dict)
    dst_device_num = dst_stage_device_num * dst_stage_num
    for rank in _progress_bar(range(dst_device_num)):
        needed_rank_list = ms.rank_list_for_transform(rank, src_strategy_dict, dst_strategy_dict)
        needed_rank_list_key = "-".join([str(r) for r in needed_rank_list])
        needed_rank_list_map[needed_rank_list_key].append(rank)
    return needed_rank_list_map


def load_file_by_param_name(filename, parme_name_list):
    result = {}
    with safe_open(filename, framework="np") as f:
        for k in parme_name_list:
            result[k] = f.get_tensor(k)
    return result


def _transform_parallel_safetensor(rank_id, param_total_dict, param_attr_dict, src_strategy_list,
                                   dst_strategy_list, param_total_dict_keys=None, src_strategy_file=None):
    """
    Transform model parallel dimension for distributed safetensor files.
    """
    transform_param_dict = {}
    device_num = -1
    param_total_dict_keys = list(param_total_dict.keys()) if param_total_dict_keys is None else param_total_dict_keys
    for param_name in param_total_dict_keys:
        tensor_shape = list(param_total_dict[param_name].values())[0].shape
        from_dev_matrix = [1]
        from_tensor_map = [-1] * len(tensor_shape)
        from_opt_shard_step = 0
        from_opt_shard_size = 0
        if src_strategy_list is not None:
            if param_name not in src_strategy_list:
                continue
            from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size = _extract_layout_item(
                src_strategy_list.get(param_name))
        to_dev_matrix_origin = [1]
        to_tensor_map_origin = [-1] * len(tensor_shape)
        to_opt_shard_step = 0
        to_opt_shard_size = 0
        if dst_strategy_list is not None:
            if param_name not in dst_strategy_list:
                continue
            to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size = _extract_layout_item(
                dst_strategy_list.get(param_name))
        # Add optimizer sharding dim for tensor layout
        device_num = np.prod(from_dev_matrix)
        if device_num < 1:
            raise ValueError("None of the parameters in safetensor file are in either src strategy or "
                             "dst strategy. Please check correctness of strategy files. "
                             "Param name is: {}, rank_id is {}.".format(param_name, rank_id))
        param_strategy = _get_tensor_strategy(from_dev_matrix, from_tensor_map)
        origin_tensor_shape = ()
        for i, item in enumerate(tensor_shape):
            if i == 0 and from_opt_shard_size > 0:
                origin_tensor_shape += (item * param_strategy[i] * from_opt_shard_size,)
                continue
            origin_tensor_shape += (item * param_strategy[i],)

        from_dev_matrix, from_tensor_map, from_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
            from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size, origin_tensor_shape)
        to_dev_matrix, to_tensor_map, to_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
            to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size, origin_tensor_shape)
        # Convert tensor layout to same device num
        from_tensor_layout, to_tensor_layout = _construct_from_to_tensor_layout(from_full_tensor_shape, from_dev_matrix,
                                                                                from_tensor_map, to_full_tensor_shape,
                                                                                to_dev_matrix, to_tensor_map)

        # when the from_layout is less devices, the safetensor_map for map[device_num] should using map[0]
        device_list = list(range(0, np.prod(from_tensor_layout[0])))
        if rank_id % device_num not in param_attr_dict[param_name] and src_strategy_file is None:
            raise ValueError("The safetensor of rank {} is missing.".format(rank_id % device_num))
        param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_tensor_layout, to_tensor_layout,
                                                                            device_list, rank_id)

        from_info_tuple = (from_opt_shard_size, from_dev_matrix, from_tensor_map, from_full_tensor_shape)
        to_info_tuple = (to_opt_shard_size, to_dev_matrix_origin, to_tensor_map_origin, origin_tensor_shape)
        _insert_opt_shard_reshape(param_rank_map, from_info_tuple, to_info_tuple)
        transform_operator_stack = _generate_transform_operator_stack(param_rank_map, rank_id)
        param_total_dict_copy = param_total_dict[param_name].copy()
        _apply_tensor_transform_operators(transform_operator_stack, param_total_dict_copy, device_num)

        transform_param_dict[param_name] = param_total_dict_copy[rank_id % device_num]

    # Handle those parameter like learning_rate, global_step which not in strategy_file.
    for param_name in param_total_dict_keys:
        if param_name not in transform_param_dict:
            transform_para = param_total_dict[param_name][rank_id % device_num]
            transform_param_dict[param_name] = transform_para
    return transform_param_dict


def unified_safetensors(src_dir, src_strategy_file, dst_dir, merge_with_redundancy=True, file_suffix=None):
    """
    Merge multiple safetensor files into a unified safetensor file.

    Args:
        src_dir (str): Source weight saving directory.
        src_strategy_file (str): Source weight segmentation strategy file.
        dst_dir (str): Target save directory.
        merge_with_redundancy (bool, optional): Whether the merged source weight files are de-duplicated and
            saved safetensors files. Default: ``True``, indicating that the merged source weight files are complete.
        file_suffix (str, optional): Specify the filename suffix for merging safetensors files. Default: ``None``,
            meaning all safetensors files in the source weight directory will be merged.

    Raises:
        ValueError: If the safetensors file of rank is missing.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> src_dir = "/usr/safetensors/llama31B/4p_safetensors/"
        >>> src_strategy_file = "/usr/safetensors/llama31B/strategy_4p.ckpt"
        >>> dst_dir = "/usr/safetensors/llama31B/merge_llama31B_4p/"
        >>> ms.unified_safetensors(src_dir, src_strategy_file, dst_dir)
    """
    _check_transform_safetensors(src_dir, "", src_strategy_file, None)
    _make_dir(dst_dir, "path")
    if os.path.isfile(src_dir):
        raise ValueError("For 'unified_safetensors', the 'src_dir' can not be a file.")
    all_safetensor_files_map = _collect_safetensor_files(src_dir, format="safetensors", file_suffix=file_suffix)
    all_ckpt_files_map = _collect_safetensor_files(src_dir, format="ckpt", file_suffix=file_suffix)
    if all_safetensor_files_map and all_ckpt_files_map:
        raise ValueError("For 'unified_safetensors', the 'src_dir' cannot contain "
                         "both ckpt file and safetensors file simultaneously")
    src_strategy_dict = _build_searched_strategy(src_strategy_file)
    src_stage_device_num = _get_device_num_from_strategy(src_strategy_dict)
    dst_stage_device_num = 1
    origin_src_strategy_list = _extract_layout_map(src_strategy_dict)
    origin_dst_strategy_list = None

    needed_rank_list_map = _find_needed_ranks(src_strategy_dict, dst_strategy_dict=None)
    for needed_rank_list, rank in needed_rank_list_map.items():
        for needed_rank in needed_rank_list.split("-"):
            if int(needed_rank) not in all_safetensor_files_map:
                raise ValueError("The safetensor file of rank{} is needed for converting rank{}'s safetensor, "
                                 "but it is missing.".format(needed_rank, rank))
    layout_map = _convert_to_list(src_strategy_dict)

    total_size = 0
    actual_params = set()
    for _, file_name in all_safetensor_files_map.items():
        total_size += os.path.getsize(file_name) / 1024 / 1024 / 1024
        with safe_open(file_name, framework="np") as f:
            actual_params.update(f.keys())
    split_num = math.ceil(total_size / 3)
    params_to_store = actual_params & set(layout_map.keys())

    name_list = []
    for name in list(params_to_store):
        if name.startswith("accu_grads"):
            continue
        name_list.append(name)
    split_list = _split_list(name_list, split_num)

    with safe_open(all_safetensor_files_map.get(0), framework="np") as f:
        all_key = f.keys()
        hyper_parameter = set(all_key) - set(name_list)
        if hyper_parameter:
            hyper_dict = {}
            for key in hyper_parameter:
                hyper_dict[key] = f.get_tensor(key)
            save_file(hyper_dict, os.path.join(dst_dir, "hyper_param.safetensors"))

    # save parameter map json
    param_name_dict = dict()
    for index, part_list in enumerate(split_list):
        for name in part_list:
            param_name_dict[name] = f"part{index}.safetensors"
    json_str = json.dumps(param_name_dict, indent=4)
    map_file = os.path.join(dst_dir, "param_name_map.json")
    with open(map_file, 'w') as f:
        f.write(json_str)

    max_process = min(split_num, 100)
    res = [i for i in range(split_num)]
    res = _split_list(res, max_process)
    processes = []
    src_strategy_name = None
    if not merge_with_redundancy:
        src_strategy_name = src_strategy_file
    for i in range(max_process):
        p = mp.Process(target=_transform_safetensors_single_semaphore, args=(
            needed_rank_list_map, all_safetensor_files_map, src_stage_device_num, dst_stage_device_num,
            src_strategy_dict, None, origin_src_strategy_list, origin_dst_strategy_list,
            "", dst_dir, "safetensors", None, split_list, res[i], True, src_strategy_name))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


def _transform_safetensors_single_semaphore(needed_rank_list_map, all_safetensor_files_map,
                                            src_stage_device_num,
                                            dst_stage_device_num,
                                            src_strategy_dict, dst_strategy_dict, origin_src_strategy_list,
                                            origin_dst_strategy_list,
                                            ckpt_prefix, dst_safetensors_dir, output_format,
                                            _transform_param_list, pipe_param_list=None, file_index=None,
                                            unified_flag=False, src_strategy_file=None):
    for i in file_index:
        _transform_safetensors_single(needed_rank_list_map, all_safetensor_files_map, src_stage_device_num,
                                      dst_stage_device_num, src_strategy_dict, dst_strategy_dict,
                                      origin_src_strategy_list,
                                      origin_dst_strategy_list, ckpt_prefix, dst_safetensors_dir, output_format,
                                      _transform_param_list, pipe_param_list[i], i, unified_flag, src_strategy_file)


def _split_list(split_list, split_num):
    split_array = np.array_split(split_list, split_num)
    return [array.tolist() for array in split_array]


def _apply_sf_obj_transform_operators(transform_operator_stack, sf_obj, device_num):
    """apply safetensors object operators"""
    if not transform_operator_stack:
        return sf_obj[:]
    level = transform_operator_stack[-1][1]
    level_operators = []
    while True:
        if not transform_operator_stack or (level != transform_operator_stack[-1][1]):
            tmp_tensor_dict = {}
            if not level_operators:
                continue
            op_name = level_operators[0][2][0]
            for operator_pair in level_operators:
                rank_id = operator_pair[0]
                cur_level = operator_pair[1]
                operator = operator_pair[2]
                if operator[0] != op_name:
                    raise ValueError("The operator in the same level should be equal in the transform tensor operator "
                                     "list, but the find {} and {} in level {}".format(op_name, operator[0], cur_level))
                if operator[0] != "AllConcat":
                    sf_obj = _apply_operator(operator[0])(sf_obj, operator)
                    continue
                for rank in operator[1][:-1]:
                    if rank % device_num not in sf_obj:
                        raise ValueError("The checkpoint file of rank {} is missing.".format(rank % device_num))
                allgather_list = [sf_obj for _ in operator[1][:-1]]
                tmp_tensor_dict[rank_id % device_num] = _apply_operator(operator[0])(allgather_list, operator)
            if op_name == "AllConcat":
                for rank, value in tmp_tensor_dict.items():
                    sf_obj = value
            level_operators.clear()
        if not transform_operator_stack:
            break
        operator_pair = transform_operator_stack.pop()
        level = operator_pair[1]
        level_operators.append(operator_pair)
    return sf_obj


def _load_parallel_checkpoint(total_safetensors_dir, dst_strategy_file, net=None, dst_safetensors_dir=None,
                              rank_id=None):
    """load parallel safetensors by merged file."""
    file_list = os.listdir(total_safetensors_dir)
    json_files = [file for file in file_list if file.endswith('.json')]
    if len(json_files) != 1:
        raise ValueError(f"For 'load_parallel_checkpoint', the number of json files in 'total_safetensors_dir' "
                         f"must be 1, but got {len(json_files)}.")
    param_name_json = os.path.join(total_safetensors_dir, json_files[0])
    with open(param_name_json, 'r') as f:
        param_name_map = json.load(f)
    if dst_strategy_file is not None:
        _, dst_strategy_list = _extract_src_dst_layout_map(rank_id, None, dst_strategy_file)
        param_list = dst_strategy_list.keys()
    else:
        dst_strategy_list = None
        param_list = param_name_map.keys()

    total_param = dict()
    dst_stage_device_num = np.prod(dst_strategy_list.get(list(dst_strategy_list.keys())[0])[0]) if dst_strategy_list \
                                                                                                   is not None else 1
    local_rank_id = rank_id % dst_stage_device_num
    for param_name in param_list:
        if param_name not in param_name_map:
            continue
        file_name = os.path.join(total_safetensors_dir, param_name_map[param_name])
        with safe_open(file_name, framework="np") as f:
            if param_name not in f.keys():
                continue
            sf_obj = f.get_slice(param_name)

        tensor_shape = sf_obj.get_shape()
        from_dev_matrix = [1]
        from_tensor_map = [-1] * len(tensor_shape)
        from_opt_shard_step = 0
        from_opt_shard_size = 0
        if dst_strategy_list is not None:
            if param_name not in dst_strategy_list:
                continue
            to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size = _extract_layout_item(
                dst_strategy_list.get(param_name))

            device_num = np.prod(from_dev_matrix)
            param_strategy = _get_tensor_strategy(from_dev_matrix, from_tensor_map)
            origin_tensor_shape = ()
            for i, item in enumerate(tensor_shape):
                if i == 0 and from_opt_shard_size > 0:
                    origin_tensor_shape += (item * param_strategy[i] * from_opt_shard_size,)
                    continue
                origin_tensor_shape += (item * param_strategy[i],)

            from_dev_matrix, from_tensor_map, from_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
                from_dev_matrix, from_tensor_map, from_opt_shard_step, from_opt_shard_size, origin_tensor_shape)
            to_dev_matrix, to_tensor_map, to_full_tensor_shape = _construct_tensor_layout_for_opt_shard(
                to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size, origin_tensor_shape)
            # Convert tensor layout to same device num
            from_tensor_layout, to_tensor_layout = _construct_from_to_tensor_layout(from_full_tensor_shape,
                                                                                    from_dev_matrix,
                                                                                    from_tensor_map,
                                                                                    to_full_tensor_shape,
                                                                                    to_dev_matrix, to_tensor_map)

            # when the from_layout is less devices, the safetensor_map for map[device_num] should using map[0]
            device_list = list(range(0, np.prod(from_tensor_layout[0])))
            param_rank_map = _get_needed_rank_transform_operator_map_by_layouts(from_tensor_layout, to_tensor_layout,
                                                                                device_list, local_rank_id)

            from_info_tuple = (from_opt_shard_size, from_dev_matrix, from_tensor_map, from_full_tensor_shape)
            to_info_tuple = (to_opt_shard_size, to_dev_matrix_origin, to_tensor_map_origin, origin_tensor_shape)
            _insert_opt_shard_reshape(param_rank_map, from_info_tuple, to_info_tuple)
            transform_operator_stack = _generate_transform_operator_stack(param_rank_map, local_rank_id)

            slice_param = _apply_sf_obj_transform_operators(transform_operator_stack, sf_obj, device_num)
        else:
            slice_param = sf_obj[:]

        total_param[param_name] = ms.Parameter(slice_param)

    if 'hyper_param.safetensors' in file_list:
        hyper_parameter_file_name = os.path.join(total_safetensors_dir, "hyper_param.safetensors")
        with safe_open(hyper_parameter_file_name, framework="np") as f:
            for key in f.keys():
                total_param[key] = ms.Parameter(f.get_tensor(key))
    if net is not None:
        param_not_load, ckpt_not_load = ms.load_param_into_net(net, total_param)
        return param_not_load, ckpt_not_load
    _make_dir(os.path.join(dst_safetensors_dir, f"rank_{rank_id}"), "path")
    ms.save_checkpoint(total_param, os.path.join(dst_safetensors_dir, f"rank_{rank_id}", f"net.safetensors"),
                       format='safetensors')
    return None


def _get_slice(rank_id, sf_obj, param_name, dst_strategy_list):
    """get slice op"""
    tensor_shape = sf_obj.get_shape()
    to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size = _extract_layout_item(
        dst_strategy_list.get(param_name))
    # Add optimizer sharding dim for tensor layout
    to_dev_matrix, to_tensor_map, _ = _construct_tensor_layout_for_opt_shard(
        to_dev_matrix_origin, to_tensor_map_origin, to_opt_shard_step, to_opt_shard_size, tensor_shape)
    slice_op = _load_tensor_shape(to_dev_matrix, to_tensor_map, full_shape=tensor_shape, rank_id=rank_id)
    shape = None
    if to_opt_shard_size > 0:
        to_tensor_strategy = _get_tensor_strategy(to_dev_matrix_origin, to_tensor_map_origin)
        to_slice_tensor_shape = ()
        for i, item in enumerate(tensor_shape):
            if i == 0 and to_opt_shard_size > 0:
                to_slice_tensor_shape += (item // (to_tensor_strategy[i] * to_opt_shard_size),)
                continue
            to_slice_tensor_shape += (item // to_tensor_strategy[i],)
        shape = list(to_slice_tensor_shape)

    return slice_op, shape


__all__ = ["_transform_safetensors", "transform_safetensors_by_stage",
           "transform_safetensors_by_rank", "ckpt_to_safetensors", "safetensors_to_ckpt", "unified_safetensors"]
