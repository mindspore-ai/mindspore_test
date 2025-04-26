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
import subprocess
from tests.st.networks.llm_parallel_feature.utils import update_parallel_speed_up_json, clear_directory


class DeepseekConfig:
    # add default config for Deepseek model.
    def __init__(self,
                 num_samples=24,
                 pp_interleave_num=2,
                 first_k_dense_replace=1,
                 num_layer=3,
                 parallel_speed_up_json=None,
                 use_gmm=True,
                 enable_deredundency=True,
                 npu_nums_per_device=2):
        # context
        self.parallel_speed_up_json = parallel_speed_up_json

        # training parameters
        self.num_samples = num_samples
        self.pp_interleave_num = pp_interleave_num

        # model parameters
        self.num_layer = num_layer
        self.first_k_dense_replace = first_k_dense_replace

        # moe
        self.use_gmm = use_gmm
        self.enable_deredundency = enable_deredundency
        self.npu_nums_per_device = npu_nums_per_device


def prepare_deepseekv3_testcase_env(testcase_name, net_config):
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    # 1. create testcase folder
    os.makedirs(os.path.join(sh_path, testcase_name), exist_ok=True)
    # 2. clear folder (if exist)
    clear_directory(f"{sh_path}/{testcase_name}")
    # 3. copy yaml to testcase folder
    os.system(f"cp {sh_path}/pretrain_deepseek3.yaml ./{testcase_name}")
    # 4. replace config in yaml
    file_path = f'{sh_path}/{testcase_name}/pretrain_deepseek3.yaml'
    status = replace_deepseekv3_config(net_config, file_path)
    # 5. update parallel_speed_up.json if needed
    if net_config.parallel_speed_up_json is not None:
        if not update_parallel_speed_up_json(testcase_name, net_config, file_path, deepseekv3=True):
            raise ValueError("Failed to update parallel_speed_up.json")
    if not status:
        raise Exception("Failed to replace config in {}".format(file_path))
    return file_path


def replace_deepseekv3_config(net_config, file_path):
    old_list = [
        "use_gmm: True",
        "num_layers 3",
        "first_k_dense_replace: 1",
        "enable_deredundency: True",
        "npu_nums_per_device: 2",
        "pp_interleave_num: 2",
    ]

    new_list = [
        f'use_gmm: {net_config.use_gmm}',
        f'num_layers {net_config.num_layer}',
        f'first_k_dense_replace: {net_config.first_k_dense_replace}',
        f'enable_deredundency: {net_config.enable_deredundency}',
        f'npu_nums_per_device: {net_config.npu_nums_per_device}',
        f'pp_interleave_num: {net_config.pp_interleave_num}',
    ]

    if len(old_list) != len(new_list):
        print(f"Old list and new list have different lengths: {len(old_list)} and {len(new_list)}")
        return False
    for i in range(len(old_list)):
        if "'" in old_list[i]:
            sed_cmd = """sed -i "s#{}#{}#g" {}""".format(old_list[i], new_list[i], file_path)
        else:
            sed_cmd = """sed -i 's#{}#{}#g' {}""".format(old_list[i], new_list[i], file_path)
        status, _ = subprocess.getstatusoutput(sed_cmd)
        if status != 0:
            print(f"Failed to replace {old_list[i]} with {new_list[i]} in {file_path}")
            return False

    # add num_samples of dataset to control the total steps
    insert_num_samples = r"sed -i '/shuffle:/a\    num_samples: {}' {}".format(net_config.num_samples, file_path)
    status, _ = subprocess.getstatusoutput(insert_num_samples)
    if status != 0:
        print(f"Failed to insert num_samples to {file_path}")
        return False

    return True
