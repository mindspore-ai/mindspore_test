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
import re
from tests.st.networks.llm_parallel_feature.utils import check_log, check_peak_memory, clear_directory
from tests.st.networks.llm_parallel_feature.deepseekv3.utils import DeepseekConfig, prepare_deepseekv3_testcase_env
from tests.mark_utils import arg_mark


def extract_losses_from_log(file_path):
    """
    Extracts all numerical values following 'loss:' from the specified log file
    and returns a list containing these values.
    :param file_path: Path to the log file
    :return: List containing all extracted loss values
    """
    # Regular expression pattern to match numerical values after "loss:"
    pattern = re.compile(r'loss:\s*([0-9]+(?:\.[0-9]+)?)')
    losses = []
    # Open the file and read its contents
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Search for matches in each line
            matches = pattern.findall(line)
            for match in matches:
                # Convert found loss values from string to float and add to list
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

    parallel_speed_up_json = {'matmul_grad_comm_overlap': 'true',
                              "pp_1f1b_overlap": "MorphAllGather,MorphReduceScatter"}
    deepseek_config = DeepseekConfig(parallel_speed_up_json=parallel_speed_up_json,
                                     use_gmm=True,
                                     enable_deredundency=True,
                                     npu_nums_per_device=2)

    file_path = prepare_deepseekv3_testcase_env(case_name, deepseek_config)

    device_num = 8
    master_port = 7123
    hccl_if_base_port = 63334

    # set env for training
    env_cmd = 'export MS_DEV_GRAPH_KERNEL_FLAGS="--enable_pass=grouped_matmul_assignadd_fusion";'
    env_cmd += 'export MS_DEV_RUNTIME_CONF="memory_statistics:True";'
    env_cmd += 'export MS_MEMORY_STATISTIC=1'

    os.system(f"{env_cmd}; bash {sh_path}/run_llm.sh {device_num} \
    {file_path} {case_name} {master_port} {hccl_if_base_port} pp")

    # check train over
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(case_name, device_num)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_peak_memory(log_path, "2700")

    # check loss
    # set the training log path
    log_file_path = f'{sh_path}/{case_name}/worker_7.log'
    # extract Training loss
    loss_list = extract_losses_from_log(log_file_path)
    # set golden_loss
    golden_loss = [13.509, 13.509, 13.507, 13.507, 13.501, 13.504]
    if_equal = golden_loss == loss_list
    assert if_equal, \
        f"Training loss is different from the golden loss, " \
        f"where training loss: {loss_list}, golden_loss: {golden_loss}."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_bmm():
    """
    Feature: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p bmm
    Description: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p bmm
    Expectation: st pass
    """
    case_name = "deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_bmm"
    sh_path = os.path.split(os.path.realpath(__file__))[0]

    parallel_speed_up_json = {'matmul_grad_comm_overlap': 'true', "pp_1f1b_overlap": "AlltoAllV,AlltoAll"}
    deepseek_config = DeepseekConfig(parallel_speed_up_json=parallel_speed_up_json,
                                     use_gmm=False,
                                     num_layer=1,
                                     pp_interleave_num=1,
                                     first_k_dense_replace=0
                                     )
    file_path = prepare_deepseekv3_testcase_env(case_name, deepseek_config)

    device_num = 8
    master_port = 7124
    hccl_if_base_port = 63355

    env_cmd = 'export MS_DEV_RUNTIME_CONF="memory_statistics:True";'
    env_cmd += 'export MS_MEMORY_STATISTIC=1'
    os.system(f"{env_cmd}; bash {sh_path}/run_llm.sh {device_num} {file_path} \
    {case_name} {master_port} {hccl_if_base_port} pp")

    # check train over
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(case_name, device_num)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_peak_memory(log_path, "2300")

    # check loss
    # set the training log path
    log_file_path = f'{sh_path}/{case_name}/worker_7.log'
    # extract Training loss
    loss_list = extract_losses_from_log(log_file_path)
    # set golden_loss
    golden_loss = [13.511, 13.504, 13.516, 13.515, 13.503, 13.508]

    if_equal = golden_loss == loss_list
    assert if_equal, \
        f"Training loss is different from the golden loss, " \
        f"where training loss: {loss_list}, golden_loss: {golden_loss}."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_gptdataset():
    """
    Feature: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p gptdataset
    Description: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p gptdataset
    Expectation: st pass
    """
    case_name = "deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_gptdataset"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    file_path = f"{sh_path}/pretrain_deepseek3_gptdataset.yaml"
    device_num = 8
    master_port = 7125
    hccl_if_base_port = 63375

    os.makedirs(os.path.join(sh_path, case_name), exist_ok=True)
    clear_directory(f"{sh_path}/{case_name}")

    env_cmd = 'export MS_DEV_RUNTIME_CONF="memory_statistics:True";'
    env_cmd += 'export MS_MEMORY_STATISTIC=1'
    os.system(f"{env_cmd};bash {sh_path}/run_llm.sh {device_num} {file_path} \
    {case_name} {master_port} {hccl_if_base_port} pp gpt")

    # check train over
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(case_name, device_num)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_peak_memory(log_path, "2700")

    # check loss
    # set the training log path
    log_file_path = f'{sh_path}/{case_name}/worker_7.log'
    # extract Training loss
    loss_list = extract_losses_from_log(log_file_path)
    # set golden_loss
    golden_loss = [12.029, 11.965, 11.790, 11.805, 11.954, 11.733]

    if_equal = golden_loss == loss_list
    assert if_equal, \
        f"Training loss is different from the golden loss, " \
        f"where training loss: {loss_list}, golden_loss: {golden_loss}."
