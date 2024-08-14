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
import re
import shutil


def clean_all_ir_files(folder_path):
    if os.path.exists(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.ir') or file_name.endswith('.dot') or \
                    file_name.endswith('.dat') or file_name.endswith('.pb'):
                os.remove(os.path.join(folder_path, file_name))


def count_ir_files_num(folder_path, ir_name):
    ir_files = list(filter(lambda f: re.match(r"%s_\d+.ir" % ir_name, f), os.listdir(folder_path)))
    return len(ir_files)


def clear_folder(folder_path):
    shutil.rmtree(folder_path)


def get_flag_from_ir_file_line(folder_path, ir_name, flag):
    ir_files = list(filter(lambda f: re.match(r"%s_\d+.ir" % ir_name, f), os.listdir(folder_path)))
    with open(r"%s/%s" % (folder_path, ir_files[-1]), 'r') as file:
        for line in file:
            if flag in line:
                value = line.split(':')[1].strip()
                return int(value)
    return None
