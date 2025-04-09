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
from tests.st.networks.llm_parallel_feature.utils import check_log, clear_directory
from tests.mark_utils import arg_mark
import re

def extract_losses_from_log(file_path):
    """
    从指定的日志文件中提取所有 'loss:' 后面跟随的数值，并返回一个包含这些数值的列表。
    :param file_path: 日志文件的路径
    :return: 包含所有提取出的损失值的列表
    """
    # 正则表达式模式匹配 "loss:" 后面跟随的数字
    pattern = re.compile(r'loss:\s*([0-9]+(?:\.[0-9]+)?)')
    losses = []
    # 打开并读取文件内容
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # 在每一行中查找匹配项
            matches = pattern.findall(line)
            for match in matches:
                # 将找到的字符串形式的损失值转换为浮点数并添加到列表中
                losses.append(float(match))
    return losses

def log_path_preprocess(case_name, device_num):
    # return the log path list, combining with rank list
    log_path_list = []
    for rank in range(device_num):
        log_path_list.append(f"./{case_name}/worker_{rank}.log")
    return log_path_list

@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_deredundency_8p_gmm():
    """
    Feature: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p gmm
    Description: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p gmm
    Expectation: st pass
    """
    case_name = "deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_deredundency_8p_gmm"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    file_path = f"{sh_path}/pretrain_deepseek3_gmm.yaml"
    device_num = 8

    os.makedirs(os.path.join(sh_path, case_name), exist_ok=True)
    clear_directory(f"{sh_path}/{case_name}")
    os.system("export MS_DEV_GRAPH_KERNEL_FLAGS='--enable_pass=grouped_matmul_assignadd_fusion'")
    os.system(f"bash {sh_path}/run_llm.sh {device_num} {file_path} {case_name} pp")

    # check train over
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(case_name, device_num)
    for log_path in real_log_path:
        check_log(log_path, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_bmm():
    """
    Feature: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p bmm
    Description: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p bmm
    Expectation: st pass
    """
    case_name = "deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_bmm"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    file_path = f"{sh_path}/pretrain_deepseek3_bmm.yaml"
    device_num = 8

    os.makedirs(os.path.join(sh_path, case_name), exist_ok=True)
    clear_directory(f"{sh_path}/{case_name}")
    os.system("export MS_DEV_GRAPH_KERNEL_FLAGS='--enable_pass=grouped_matmul_assignadd_fusion'")
    os.system(f"bash {sh_path}/run_llm.sh {device_num} {file_path} {case_name} pp")

    # check train over
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(case_name, device_num)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
