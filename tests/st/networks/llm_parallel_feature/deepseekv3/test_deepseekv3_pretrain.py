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
import numpy as np
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


def extract_average_step_time_from_log(file_path):
    """
    Extracts all numerical values following 'per_step_time:' from the specified log file
    and returns the average step time calculated by per step time.
    :param file_path: Path to the log file
    :return: Average time
    """
    # Regular expression pattern to match numerical values after "per_step_time:"
    pattern = re.compile(r'per_step_time:\s*([0-9]+(?:\.[0-9]+)?)')
    per_step_time_list = []

    # Open the file and read its contents
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Search for matches in each line
            matches = pattern.findall(line)
            for match in matches:
                # Convert found step time values from string to float and add to list
                per_step_time_list.append(float(match))

    # del the first two values.
    per_step_time_list = per_step_time_list[2:]

    # calculate the mean step time.
    average_time = np.mean(per_step_time_list)
    return average_time


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

    parallel_speed_up_json = {'matmul_grad_comm_overlap': True,
                              "pp_1f1b_overlap": "MorphAllGather,MorphReduceScatter"}

    deepseek_config = DeepseekConfig(parallel_speed_up_json=parallel_speed_up_json,
                                     num_samples=60,
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
        # self-test results: 2614M, memory should be lower than 2614+50=2664M
        check_peak_memory(log_path, "2664")

    # check loss
    # set the training log path
    log_file_path = f'{sh_path}/{case_name}/worker_7.log'

    # extract Training loss
    loss_list = extract_losses_from_log(log_file_path)

    # set golden_loss
    golden_loss = [13.509, 13.509, 13.507, 13.507, 13.501, \
                   13.503, 13.503, 13.497, 13.485, 13.494, \
                   13.486, 13.478, 13.475, 13.461, 13.457]
    if_equal = np.allclose(
        np.array(golden_loss), np.array(loss_list), atol=1e-3, rtol=0
    )
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

    parallel_speed_up_json = {'matmul_grad_comm_overlap': True, "pp_1f1b_overlap": "AlltoAllV,AlltoAll"}
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
        # self-test results: 2182M, memory should be lower than 2182+50=2232M
        check_peak_memory(log_path, "2232")

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
        # self-test results: 2614M, memory should be lower than 2614+50=2664M
        check_peak_memory(log_path, "2664")

    # check loss
    # set the training log path
    log_file_path = f'{sh_path}/{case_name}/worker_7.log'

    # extract training loss
    loss_list = extract_losses_from_log(log_file_path)

    # set golden_loss
    golden_loss = [12.029, 11.965, 11.790, 11.805, 11.954, 11.733]

    if_equal = golden_loss == loss_list
    assert if_equal, \
        f"Training loss is different from the golden loss, " \
        f"where training loss: {loss_list}, golden_loss: {golden_loss}."

    # check per step time
    # self-test results: 170ms, step time should be lower than 170+20=190ms
    excepted_average_step_time = 200

    # extract training step time
    average_step_time = extract_average_step_time_from_log(log_file_path)

    # check if the step time is lower than the excepted_average_step_time
    step_time_pass = excepted_average_step_time > average_step_time
    assert step_time_pass, \
        f"Training average step time is larger than the excepted average step time," \
        f"where training average step time is {average_step_time},  excepted step time is {excepted_average_step_time}."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_gmm_performance():
    """
    Feature: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p gmm performance
    Description: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p gmm performance
    Expectation: st pass
    """
    case_name = "deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_gmm_performance"
    sh_path = os.path.split(os.path.realpath(__file__))[0]

    # set the speed up json
    parallel_speed_up_json = {'matmul_grad_comm_overlap': True}

    # set the config
    deepseek_config = DeepseekConfig(num_samples=24,
                                     hidden_size=4096,
                                     intermediate_size=8192,
                                     moe_intermediate_size=2048,
                                     parallel_speed_up_json=parallel_speed_up_json,
                                     use_gmm=True,
                                     enable_deredundency=False,
                                     npu_nums_per_device=2,
                                     use_fused_ops_permute=True,
                                     use_fused_swiglu=True,
                                     enable_fa_var_len=True,
                                     use_fused_rope=True,
                                     pp_interleave_num=1,
                                     deterministic="OFF"
                                     )

    file_path = prepare_deepseekv3_testcase_env(case_name, deepseek_config)

    # set the communication parameters
    device_num = 8
    master_port = 7124
    hccl_if_base_port = 63395

    # set env for training
    graph_kernel_flags = "--enable_pass=grouped_matmul_assignadd_fusion " \
                         "--enable_cluster_ops=MatMul,BatchMatMul,Reshape --online_tuning=1"

    os.system(f"bash {sh_path}/run_llm.sh {device_num} \
    {file_path} {case_name} {master_port} {hccl_if_base_port} pp mindrecord \"{graph_kernel_flags}\"")

    # check train over
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(case_name, device_num)
    for log_path in real_log_path:
        check_log(log_path, check_pair)

    # check per step time
    # set the training log path
    log_file_path = f'{sh_path}/{case_name}/worker_7.log'

    # self-test results: 264ms, step time should be lower than 264+30=294ms
    excepted_average_step_time = 294

    # extract training step time
    average_step_time = extract_average_step_time_from_log(log_file_path)

    # check if the step time is lower than the excepted_average_step_time
    step_time_pass = excepted_average_step_time > average_step_time
    assert step_time_pass, \
        f"Training average step time is larger than the excepted average step time," \
        f"where training average step time is {average_step_time}, " \
        f"excepted step time is {excepted_average_step_time}."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_1b1f_performance():
    """
    Feature: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p gmm 1b1f performance
    Description: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p gmm 1b1f performance
    Expectation: st pass
    """
    case_name = "deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_1b1f_performance"
    sh_path = os.path.split(os.path.realpath(__file__))[0]

    # set the speed up json
    parallel_speed_up_json = {'matmul_grad_comm_overlap': True,
                              'pp_1f1b_overlap': 'AlltoAllV,AlltoAll'}

    # set the config
    deepseek_config = DeepseekConfig(num_samples=24,
                                     hidden_size=4096,
                                     intermediate_size=8192,
                                     moe_intermediate_size=2048,
                                     parallel_speed_up_json=parallel_speed_up_json,
                                     use_gmm=True,
                                     enable_deredundency=False,
                                     npu_nums_per_device=2,
                                     use_fused_ops_permute=True,
                                     use_fused_swiglu=True,
                                     enable_fa_var_len=True,
                                     use_fused_rope=True,
                                     pp_interleave_num=2,
                                     deterministic="OFF"
                                     )

    file_path = prepare_deepseekv3_testcase_env(case_name, deepseek_config)

    # set the communication parameters
    device_num = 8
    master_port = 7125
    hccl_if_base_port = 63415

    # set env for training
    graph_kernel_flags = "--enable_pass=grouped_matmul_assignadd_fusion " \
                         "--enable_cluster_ops=MatMul,BatchMatMul,Reshape --online_tuning=1"

    os.system(f"bash {sh_path}/run_llm.sh {device_num} \
    {file_path} {case_name} {master_port} {hccl_if_base_port} pp mindrecord \"{graph_kernel_flags}\"")

    # check train over
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(case_name, device_num)
    for log_path in real_log_path:
        check_log(log_path, check_pair)

    # check per step time
    # set the training log path
    log_file_path = f'{sh_path}/{case_name}/worker_7.log'

    # set the excepted average step time
    # self-test results: 265ms, step time should be lower than 265+30=295ms
    excepted_average_step_time = 295

    # extract training step time
    average_step_time = extract_average_step_time_from_log(log_file_path)

    # check if the step time is lower than the excepted_average_step_time
    step_time_pass = excepted_average_step_time > average_step_time

    assert step_time_pass, \
        f"Training average step time is larger than the excepted average step time," \
        f"where training average step time is {average_step_time},  excepted step time is {excepted_average_step_time}."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='allcards', essential_mark='essential')
def test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_bmm_performance():
    """
    Feature: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p bmm performance
    Description: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 8p bmm performance
    Expectation: st pass
    """
    case_name = "deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_performance_8p_bmm_performance"
    sh_path = os.path.split(os.path.realpath(__file__))[0]

    # set the speed up json
    parallel_speed_up_json = {'matmul_grad_comm_overlap': True,
                              'pp_1f1b_overlap': 'AlltoAllV,AlltoAll'}

    # set the config
    deepseek_config = DeepseekConfig(num_samples=24,
                                     hidden_size=4096,
                                     intermediate_size=8192,
                                     moe_intermediate_size=2048,
                                     parallel_speed_up_json=parallel_speed_up_json,
                                     use_gmm=False,
                                     use_fused_swiglu=True,
                                     enable_fa_var_len=True,
                                     use_fused_rope=True,
                                     pp_interleave_num=2,
                                     deterministic="OFF"
                                     )

    file_path = prepare_deepseekv3_testcase_env(case_name, deepseek_config)

    # set the communication parameters
    device_num = 8
    master_port = 7126
    hccl_if_base_port = 63435

    # set env for training
    graph_kernel_flags = "--enable_cluster_ops=MatMul,BatchMatMul,Reshape --online_tuning=1"

    os.system(f"bash {sh_path}/run_llm.sh {device_num} \
    {file_path} {case_name} {master_port} {hccl_if_base_port} pp mindrecord \"{graph_kernel_flags}\"")

    # check train over
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(case_name, device_num)
    for log_path in real_log_path:
        check_log(log_path, check_pair)

    # check per step time
    # set the training log path
    log_file_path = f'{sh_path}/{case_name}/worker_7.log'

    # set the excepted average step time
    # self-test results: 262ms, step time should be lower than 262+30=292ms
    excepted_average_step_time = 292

    # extract training step time
    average_step_time = extract_average_step_time_from_log(log_file_path)

    # check if the step time is lower than the excepted_average_step_time
    step_time_pass = excepted_average_step_time > average_step_time

    assert step_time_pass, \
        f"Training average step time is larger than the excepted average step time," \
        f"where training average step time is {average_step_time},  excepted step time is {excepted_average_step_time}."


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='essential')
def test_deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_mte_8p_gptdataset():
    """
    Feature: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 mte 8p gptdataset
    Description: test deepseekv3 cell dp2mp2ep4pp2mb4gas1bs1 mte 8p gptdataset
    Expectation: st pass
    """
    case_name = "deepseekv3_cell_dp2mp2ep2pp2mb4gas1bs1_8p_gptdataset"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    file_path = f"{sh_path}/pretrain_deepseek3_mte_gptdataset.yaml"
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
        # self-test results: 745M, memory should be lower than 745 * 110%=820M
        check_peak_memory(log_path, "820")

    # check loss
    # set the training log path
    log_file_path = f'{sh_path}/{case_name}/worker_7.log'

    # extract training loss
    loss_list = extract_losses_from_log(log_file_path)

    # set golden_loss
    golden_loss = [12.029, 11.965, 11.790, 11.805, 11.954, 11.733]

    if_equal = golden_loss == loss_list
    assert if_equal, \
        f"Training loss is different from the golden loss, " \
        f"where training loss: {loss_list}, golden_loss: {golden_loss}."

    # check per step time
    # self-test results: 170ms, step time should be lower than 170+20=190ms
    excepted_average_step_time = 200

    # extract training step time
    average_step_time = extract_average_step_time_from_log(log_file_path)

    # check if the step time is lower than the excepted_average_step_time
    step_time_pass = excepted_average_step_time > average_step_time
    assert step_time_pass, \
        f"Training average step time is larger than the excepted average step time," \
        f"where training average step time is {average_step_time},  excepted step time is {excepted_average_step_time}."
