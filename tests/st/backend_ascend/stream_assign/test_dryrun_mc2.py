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
"""
Test module for stream assign.
"""
import os
import re
from tests.mark_utils import arg_mark


def extract_stream_id(log_line):
    pattern = r'stream id\[(\d+)\]'
    match = re.search(pattern, log_line)

    if match:
        stream_id = match.group(0)
        number = match.group(1)
        return stream_id, number
    return None, None


def extract_group(log_line):
    pattern = r'group\[([^\]]+)\]'
    match = re.search(pattern, log_line)

    if match:
        group = match.group(0)
        group_id = match.group(1)
        return group, group_id
    return None, None


def run_command(cmd, log_path):
    if os.path.isfile(log_path):
        os.remove(log_path)
    os.system(cmd)
    order = []
    with open(log_path, 'r') as file:
        for line in file:
            has_exec_order = ('PrintGraphExecuteOrder' in line)
            has_hccl_op = any(keyword in line for keyword in ['AllGatherMatmul', 'MatmulReduceScatter', 'AllReduce'])
            if has_exec_order and has_hccl_op:
                order.append(line)
    stream_dict = {}
    for line in order:
        print(line)
        _, number = extract_stream_id(line)
        _, group_id = extract_group(line)
        assert number != "0"
        if group_id not in stream_dict:
            stream_dict[group_id] = number
        else:
            assert stream_dict[group_id] == number
    print(stream_dict)
    assert len(stream_dict) == 3
    values_set = set(stream_dict.values())
    assert len(values_set) == 3
    if os.path.isfile(log_path):
        os.remove(log_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_mc2():
    """
    Feature: stream assign
    Description: test mc2 stream assign by group
    Expectation: run success
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/dryrun_mc2.sh", f"{sh_path}/mc2.log")
