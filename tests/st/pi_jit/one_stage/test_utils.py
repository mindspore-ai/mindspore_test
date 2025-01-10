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
"""Test utils for one stage"""
import os
import glob
import shutil


def count_file_key(file, key):
    """Count key string in file"""
    appear_count = 0
    with open(file, 'r') as fp:
        for line in fp:
            if key in line:
                appear_count += 1
    return appear_count


def check_ir_info(func, inputs, expect_dict, expect_file, expect_num, target_dir):
    """After func run with input, check whether create expect_num of expect_file in target_dir match expect_dict"""
    os.environ['MS_DEV_SAVE_GRAPHS'] = '1'
    os.environ['MS_DEV_SAVE_GRAPHS_PATH'] = target_dir
    os.environ['MS_DEV_DUMP_IR_PASSES'] = expect_file
    try:
        func(*inputs)
        ir_files = glob.glob(os.path.join(target_dir, "*" + expect_file + "*.ir"))
        assert len(ir_files) == expect_num
        for key in expect_dict:
            real_count = 0
            for file in ir_files:
                real_count += count_file_key(file, key)
            assert real_count == expect_dict[key]
    finally:
        os.unsetenv('MS_DEV_SAVE_GRAPHS')
        os.unsetenv('MS_DEV_SAVE_GRAPHS_PATH')
        os.unsetenv('MS_DEV_DUMP_IR_PASSES')
        shutil.rmtree(target_dir)
