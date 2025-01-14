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

# 1. copy mindformers to corresponding folder
# 2. replace parts of value in yaml through replace_config func
# 3. run st in dryrun mode

import os
from tests.st.networks.llm_parallel_feature.utils import prepare_mixtral_testcase_env, check_log, MixtralConfig, \
    log_path_preprocess

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_mixtral_cell_dp2mp4ep2pp2mb4gas1bs1_16p():
    """
    Feature: test mixtral_8x7b cell dp2mp4ep2pp2mb4gas1bs1 16p
    Description: test mixtral_8x7b cell dp2mp4ep2pp2mb4gas1bs1 16p
    Expectation: st pass
    """
    case_name = "mixtral_cell_dp2mp4ep2pp2mb4gas1bs1_16p"
    rank_list = "0"
    mixtral_config = MixtralConfig(case_name=case_name,
                                   data_parallel=2,
                                   model_parallel=4,
                                   pipeline_stage=2,
                                   expert_parallel=2,
                                   micro_batch_num=2,
                                   enable_parallel_optimizer=True,
                                   vocab_emb_dp=True,
                                   recompute=True,
                                   gradient_accumulation_steps=4,
                                   group_wise_a2a=True,
                                   use_fused_ops_topkrouter=True
                                   )
    output_file, file_path = prepare_mixtral_testcase_env(case_name, mixtral_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_mixtral_cell_dp4mp2ep4pp2mb2gas1bs1_seqp_16p():
    """
    Feature: test mixtral_8x7b cell dp4mp2ep4pp2mb2gas1bs1_seqp 16p
    Description: test mixtral_8x7b cell dp4mp2ep4pp2mb2gas1bs1_seqp 16p
    Expectation: st pass
    """
    case_name = "mixtral_cell_dp4mp2ep4pp2mb2gas1bs1_seqp_16p"
    rank_list = "0"
    mixtral_config = MixtralConfig(case_name=case_name,
                                   data_parallel=4,
                                   model_parallel=2,
                                   pipeline_stage=2,
                                   expert_parallel=4,
                                   micro_batch_num=2,
                                   enable_parallel_optimizer=True,
                                   use_seq_parallel=True,
                                   vocab_emb_dp=True,
                                   recompute=True,
                                   gradient_accumulation_steps=2
                                   )
    output_file, file_path = prepare_mixtral_testcase_env(case_name, mixtral_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_mixtral_cell_dp1mp4ep1pp4mb8gas1bs1_16p():
    """
    Feature: test mixtral_8x7b cell dp1mp4ep1pp4mb8gas1bs1 16p
    Description: test mixtral_8x7b cell dp1mp4ep1pp4mb8gas1bs1 16p
    Expectation: st pass
    """
    case_name = "mixtral_cell_dp1mp4ep1pp4mb8gas1bs1_16p"
    rank_list = "0"
    mixtral_config = MixtralConfig(case_name=case_name,
                                   data_parallel=1,
                                   model_parallel=4,
                                   pipeline_stage=4,
                                   expert_parallel=1,
                                   micro_batch_num=4,
                                   enable_parallel_optimizer=False,
                                   vocab_emb_dp=False,
                                   recompute=True,
                                   gradient_accumulation_steps=8
                                   )
    output_file, file_path = prepare_mixtral_testcase_env(case_name, mixtral_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_mixtral_dp2mp2ep2pp2mb4gas1bs1_8p():
    """
    Feature: test mixtral_8x7b dp2mp2ep2pp2mb4gas1bs1 8p
    Description: test mixtral_8x7b dp2mp2ep2pp2mb4gas1bs1 8p
    Expectation: st pass
    """
    case_name = "mixtral_dp2mp2ep2pp2mb4gas1bs1_8p"
    rank_list = "0"
    mixtral_config = MixtralConfig(case_name=case_name,
                                   data_parallel=2,
                                   model_parallel=2,
                                   pipeline_stage=2,
                                   expert_parallel=2,
                                   micro_batch_num=2,
                                   enable_parallel_optimizer=True,
                                   vocab_emb_dp=False,
                                   recompute=True,
                                   gradient_accumulation_steps=4,
                                   group_wise_a2a=True
                                   )
    output_file, file_path = prepare_mixtral_testcase_env(case_name, mixtral_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name} no_pp")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_mixtral_cell_dp4mp1ep2pp2mb2gas1bs1_8p():
    """
    Feature: test mixtral_8x7b cell dp4mp1ep2pp2mb2gas1bs1 8p
    Description: test mixtral_8x7b cell dp4mp1ep2pp2mb2gas1bs1 8p
    Expectation: st pass
    """
    case_name = "mixtral_cell_dp4mp1ep2pp2mb2gas1bs1_8p"
    rank_list = "0"
    mixtral_config = MixtralConfig(case_name=case_name,
                                   data_parallel=4,
                                   model_parallel=1,
                                   pipeline_stage=2,
                                   expert_parallel=2,
                                   micro_batch_num=2,
                                   enable_parallel_optimizer=True,
                                   vocab_emb_dp=False,
                                   recompute=True,
                                   select_recompute=True,
                                   gradient_accumulation_steps=2,
                                   group_wise_a2a=True,
                                   use_fused_ops_topkrouter=True
                                   )
    output_file, file_path = prepare_mixtral_testcase_env(case_name, mixtral_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name} no_pp")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_mixtral_dp8mp1ep8pp1mb1gas1bs1_8p():
    """
    Feature: test mixtral_8x7b dp8mp1ep8pp1mb1gas1bs1 8p
    Description: test mixtral_8x7b dp8mp1ep8pp1mb1gas1bs1 8p
    Expectation: st pass
    """
    case_name = "mixtral_dp8mp1ep8pp1mb1gas1bs1_8p"
    rank_list = "0"
    mixtral_config = MixtralConfig(case_name=case_name,
                                   data_parallel=8,
                                   model_parallel=1,
                                   pipeline_stage=1,
                                   expert_parallel=1,
                                   micro_batch_num=1,
                                   enable_parallel_optimizer=True,
                                   vocab_emb_dp=False,
                                   recompute=True,
                                   gradient_accumulation_steps=1,
                                   group_wise_a2a=True,
                                   use_fused_ops_topkrouter=True
                                   )
    output_file, file_path = prepare_mixtral_testcase_env(case_name, mixtral_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name} no_pp")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_mixtral_dp4m1ep2pp2mb2gas1bs1_8p():
    """
    Feature: test mixtral_8x7b dp4m1ep2pp2mb2gas1bs1 8p
    Description: test mixtral_8x7b dp4m1ep2pp2mb2gas1bs1 8p
    Expectation: st pass
    """
    case_name = "mixtral_dp4m1ep2pp2mb2gas1bs1_8p"
    rank_list = "0"
    mixtral_config = MixtralConfig(case_name=case_name,
                                   data_parallel=4,
                                   model_parallel=1,
                                   pipeline_stage=2,
                                   expert_parallel=2,
                                   micro_batch_num=2,
                                   enable_parallel_optimizer=True,
                                   vocab_emb_dp=False,
                                   recompute=True,
                                   gradient_accumulation_steps=2
                                   )
    output_file, file_path = prepare_mixtral_testcase_env(case_name, mixtral_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name}")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_mixtral_dp8m1ep8pp1mb1gas1bs1_8p():
    """
    Feature: test mixtral_8x7b dp8m1ep8pp1mb1gas1bs1 8p
    Description: test mixtral_8x7b dp8m1ep8pp1mb1gas1bs1 8p
    Expectation: st pass
    """
    case_name = "mixtral_dp8m1ep8pp1mb1gas1bs1_8p"
    rank_list = "0"
    mixtral_config = MixtralConfig(case_name=case_name,
                                   data_parallel=8,
                                   model_parallel=1,
                                   pipeline_stage=1,
                                   expert_parallel=8,
                                   micro_batch_num=1,
                                   enable_parallel_optimizer=False,
                                   vocab_emb_dp=True,
                                   recompute=True,
                                   gradient_accumulation_steps=1
                                   )
    output_file, file_path = prepare_mixtral_testcase_env(case_name, mixtral_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name}")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
