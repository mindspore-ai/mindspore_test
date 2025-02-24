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
import pytest
from tests.st.networks.llm_parallel_feature.utils import prepare_testcase_env, check_log, LLMConfig, \
    log_path_preprocess, graph_path_preprocess, check_peak_memory, check_compile_time, check_graph, \
    check_node_strategy, check_param_shape, check_comm_op_groups, \
    find_graph_file_name

from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_dp2mp4pp1_recompute():
    """
    Feature: test llama2 dp2mp4pp1 full_recompute
    Description: test llama2 dp2mp4pp1 full_recompute
    Expectation: st pass
    """
    case_name = "llama2_dp2mp4pp1_recompute"
    rank_list = "0"
    llama2_config = LLMConfig(case_name=case_name, data_parallel=2, model_parallel=4,
                              enable_parallel_optimizer=False, batch_size=4, vocab_emb_dp=False,
                              parallel_speed_up_json={
                                  'matmul_grad_comm_overlap': 'true'})
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name}")
    check_pair = {"Training Over": 1}
    real_graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    graph_path = real_graph_path[0]
    attrs_check_pairs = {"recompute: Bool(1)": 18}
    validate_name = find_graph_file_name(graph_path, "validate")
    check_graph(graph_path, validate_name, attrs_check_pairs)
    gather_strategy_check_pairs = {"PrimFunc_Gather": {"": "((4, 1), (2, 1))"}}
    check_node_strategy(graph_path, validate_name, gather_strategy_check_pairs)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_peak_memory(log_path, "10300")
        check_compile_time(log_path, 15)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_dp4mp4pp1op_recompute():
    """
    Feature: test llama2 dp4mp4pp1op full_recompute
    Description: test llama2 dp4mp4pp1op full_recompute
    Expectation: st pass
    """
    case_name = "llama2_dp4mp4pp1op_recompute"
    rank_list = "0"
    # wait for fixing
    llama2_config = LLMConfig(case_name=case_name, data_parallel=4, model_parallel=4,
                              recompute=True, batch_size=2, vocab_emb_dp=False,
                              parallel_speed_up_json={
                                  'enable_grad_comm_opt': 'true',
                                  'enable_opt_shard_comm_opt': 'true'})

    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name}")
    check_pair = {"Training Over": 1}
    real_graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    validate_name = find_graph_file_name(real_graph_path[0], "validate")
    hwopt_after_inline_name = find_graph_file_name(real_graph_path[0], "hwopt_d_after_inline_graph_0")
    graph_path = real_graph_path[0]
    attrs_check_pairs = {", recompute: Bool(1)": 207}
    check_graph(graph_path, validate_name, attrs_check_pairs)
    param_parallel_speed_up_check_pairs = {'last_grad_comm_compute_depend: Bool(1)': '39',
                                           'grad_comm_dx_depend: Bool(1)': '1'}
    check_graph(graph_path, hwopt_after_inline_name, param_parallel_speed_up_check_pairs)
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_peak_memory(log_path, "3900")
        check_compile_time(log_path, 15)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_cell_dp2mp4pp1op_grad_accu():
    """
    Feature: test llama2 cell_dp2mp4pp1op_grad_accu
    Description: test llama2 cell_dp2mp4pp1op_grad_accu
    Expectation: st pass
    """
    case_name = "llama2_cell_dp2mp4pp1op_grad_accu"
    rank_list = "0"
    # wait for fixing
    llama2_config = LLMConfig(case_name=case_name, data_parallel=2, model_parallel=4,
                              gradient_accumulation_steps=4, batch_size=1,
                              recompute=False,
                              parallel_speed_up_json={
                                  'enable_grad_comm_opt': 'false',
                                  'enable_opt_shard_comm_opt': 'false'})
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name} no_pp")
    check_pair = {"Training Over": 1}
    ops_check_pairs = {"VirtualAssignAdd": 39}
    graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)[0]
    validate_name = find_graph_file_name(graph_path, "validate")
    step_parallel_end_name = find_graph_file_name(graph_path, "step_parallel_end")
    check_graph(graph_path, step_parallel_end_name, ops_check_pairs)
    gather_strategy_check_pairs = {"PrimFunc_Gather": {"298_output": "((1, 1), (2, 1))"}}
    check_node_strategy(graph_path, validate_name, gather_strategy_check_pairs)
    param_opt_shape_check_pairs = {"_model.layers.0.attention.wq.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wk.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wv.weight": "(512, 4096)",
                                   "accu_grads.model.layers.0.attention.wq.weight": "(1024, 4096)",
                                   "accu_grads.model.layers.0.attention.wk.weight": "(1024, 4096)",
                                   "accu_grads.model.layers.0.attention.wv.weight": "(1024, 4096)",
                                   '_adam_m.model.layers.0.attention.wq.weight': '(512, 4096)',
                                   '_adam_m.model.layers.0.attention.wk.weight': '(512, 4096)',
                                   '_adam_m.model.layers.0.attention.wv.weight': '(512, 4096)'}
    # param1_dependency_list = ["CNode_198", 0, "_MicroStepAllGather"]
    # param2_dependency_list = ["CNode_198", 4, "_MicroStepAllGather"]
    # check_node_dependency_backward_search(graph_path, step_parallel_end_name, 150, param1_dependency_list)
    # check_node_dependency_backward_search(graph_path, step_parallel_end_name, 150, param2_dependency_list)
    check_param_shape(graph_path, step_parallel_end_name, 100, param_opt_shape_check_pairs)
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_peak_memory(log_path, "7200")
        check_compile_time(log_path, 15)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_cell_dp2mp4pp2vpp4op_1f1b():
    """
    Feature: test llama2_cell_dp2mp4pp2vpp4op_1f1b
    Description: test llama2_cell_dp2mp4pp2vpp4op_1f1b
    Expectation: st pass
    """
    case_name = "llama2_cell_dp2mp4pp2vpp4op_1f1b"
    rank_list = "0,8"
    # wait for fixing
    llama2_config = LLMConfig(case_name, data_parallel=2, model_parallel=4, pipeline_stage=2,
                              micro_batch_num=2, batch_size=2, pp_interleave_num=4,
                              pipeline_interleave=True, pipeline_scheduler="1f1b",
                              num_layers=8, recompute=False,
                              parallel_speed_up_json={
                                  'enable_grad_comm_opt': 'false',
                                  'enable_opt_shard_comm_opt': 'false'})
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    # stage 0
    ops_check_pairs_0 = {"VirtualAssignAdd": 74}
    validate_name = find_graph_file_name(graph_path[0], "validate")
    step_parallel_end_name = find_graph_file_name(graph_path[0], "step_parallel_end")
    gather_strategy_check_pairs = {"PrimFunc_Gather": {"": "((4, 1), (2, 1))"}}
    param_opt_shape_check_pairs = {"_model.layers.0.attention.wq.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wk.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wv.weight": "(512, 4096)",
                                   "accu_grads.model.layers.0.attention.wq.weight": "(1024, 4096)",
                                   "accu_grads.model.layers.0.attention.wk.weight": "(1024, 4096)",
                                   "accu_grads.model.layers.0.attention.wv.weight": "(1024, 4096)",
                                   '_adam_m.model.layers.0.attention.wq.weight': '(512, 4096)',
                                   '_adam_m.model.layers.0.attention.wk.weight': '(512, 4096)',
                                   '_adam_m.model.layers.0.attention.wv.weight': '(512, 4096)'}
    check_graph(graph_path[0], step_parallel_end_name, ops_check_pairs_0)
    check_param_shape(graph_path[0], validate_name, 100, param_opt_shape_check_pairs)
    check_node_strategy(graph_path[0], validate_name, gather_strategy_check_pairs)
    # dependency_list_0_0 = ["CNode_1228", 0, "_MicroStepAllGather"]
    # dependency_list_0_1 = ["CNode_1228", 3, "_MicroStepAllGather"]
    # check_node_dependency_backward_search(graph_path[0], step_parallel_end_name, 100, dependency_list_0_0)
    # check_node_dependency_backward_search(graph_path[0], step_parallel_end_name, 100, dependency_list_0_1)
    # stage 1
    step_parallel_end_name_1 = find_graph_file_name(graph_path[1], "step_parallel_end")
    ops_check_pairs_1 = {"VirtualAssignAdd": 76}
    check_graph(graph_path[1], step_parallel_end_name_1, ops_check_pairs_1)
    check_param_shape(graph_path[0], validate_name, 100, param_opt_shape_check_pairs)
    # dependency_list_1_0 = ["CNode_1237", 0, "_MicroStepAllGather"]
    # dependency_list_1_1 = ["CNode_1237", 3, "_MicroStepAllGather"]
    # check_node_dependency_backward_search(graph_path[1], step_parallel_end_name_1, 100, dependency_list_1_0)
    # check_node_dependency_backward_search(graph_path[1], step_parallel_end_name_1, 100, dependency_list_1_1)
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_compile_time(log_path, 15)
    check_peak_memory(real_log_path[0], "8200")
    check_peak_memory(real_log_path[1], "8200")


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute():
    """
    Feature: test lama2_cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute
    Description: test lama2_cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute
    Expectation: st pass
    """
    case_name = "llama2_cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute"
    rank_list = "0,8"
    # wait for fixing
    llama2_config = LLMConfig(case_name, data_parallel=2, model_parallel=1, pipeline_stage=2,
                              micro_batch_num=4, batch_size=1, pp_interleave_num=2,
                              pipeline_interleave=True, pipeline_scheduler="1f1b",
                              num_layers=4, context_parallel=4, select_recompute=True,
                              recompute=False, enable_parallel_optimizer=False,
                              parallel_speed_up_json={
                                  'enable_grad_comm_opt': 'true',
                                  'enable_opt_shard_comm_opt': 'true'})
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    # stage0
    validate_name = find_graph_file_name(graph_path[0], "validate")
    hwopt_after_inline_name = find_graph_file_name(graph_path[0], "hwopt_d_after_inline_graph_0")
    attrs_check_pairs = {" recompute: Bool(1)": 7}
    check_graph(graph_path[0], validate_name, attrs_check_pairs)
    param_parallel_speed_up_check_pairs = {'grad_comm_assign_add_depend: Bool(1)': '19',
                                           'last_grad_comm_compute_depend: Bool(1)': '19',
                                           'grad_comm_dx_depend': '14'}
    check_graph(graph_path[0], hwopt_after_inline_name, param_parallel_speed_up_check_pairs)

    # stage1
    validate_name_1 = find_graph_file_name(graph_path[1], "validate")
    attrs_check_pairs_1 = {" recompute: Bool(1)": 10}
    check_graph(graph_path[1], validate_name_1, attrs_check_pairs_1)
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_compile_time(log_path, 15)
    check_peak_memory(real_log_path[0], "9300")
    check_peak_memory(real_log_path[1], "10300")




@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_cell_dp2mp1pp2vpp2cpring_1f1b_recompute():
    """
    Feature: test llama2 cell_dp2mp1pp2vpp2cpring_1f1b_recompute
    Description: test llama2 cell_dp2mp1pp2vpp2cpring_1f1b_recompute
    Expectation: st pass
    """
    case_name = "llama2_cell_dp2mp1pp2vpp2cpring_1f1b_recompute"
    rank_list = "0,8"
    # wait for fixing
    llama2_config = LLMConfig(case_name=case_name,
                              data_parallel=2,
                              model_parallel=1,
                              pipeline_stage=2, micro_batch_num=2,
                              pp_interleave_num=2, pipeline_interleave=True, pipeline_scheduler="1f1b",
                              batch_size=1,
                              context_parallel=4,
                              use_ring_attention=True,
                              parallel_speed_up_json={
                                  'enable_grad_comm_opt': 'false',
                                  'enable_opt_shard_comm_opt': 'false'},
                              recompute=True)
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")

    real_graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    graph_path = real_graph_path[0]
    validate_name = find_graph_file_name(graph_path, "validate")
    step_parallel_end_name = find_graph_file_name(graph_path, "step_parallel_end")
    # op 权重切分
    param_opt_shape_check_pairs = {"_model.layers.0.attention.wq.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wk.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wv.weight": "(512, 4096)"}
    check_param_shape(graph_path, step_parallel_end_name, 100, param_opt_shape_check_pairs)
    # op load 前有 allgather
    # param1_dependency_list = ['CNode_753', 0, '_MicroStepAllGather']
    # check_node_dependency_backward_search(graph_path, step_parallel_end_name, 200, param1_dependency_list)
    # recompute
    attrs_check_pairs = {" recompute: Bool(1)": '8'}
    check_graph(graph_path, validate_name, attrs_check_pairs)
    # dp、mp Gather 切分
    gather_strategy_check_pairs = {"PrimFunc_Gather": {"%0, %5": "((4, 1), (2, 1))"}}
    check_node_strategy(graph_path, validate_name, gather_strategy_check_pairs)

    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    log_path = None
    for log_path in real_log_path:
        check_log(log_path, check_pair)
    check_peak_memory(log_path, "5000")
    check_compile_time(log_path, 15)


@pytest.mark.skip(reason="has bug need fix")
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_cell_dp2mp1pp2vpp2cpulysse_1f1b_select_recompute():
    """
    Feature: test llama2 cell_dp2mp1pp2vpp2cpulysse_1f1b_select_recompute
    Description: test llama2 cell_dp2mp1pp2vpp2cpulysse_1f1b_select_recompute
    Expectation: st pass
    """
    case_name = "llama2_cell_dp2mp1pp2vpp2cpulysse_1f1b_select_recompute"
    rank_list = "0,8"
    # wait for fixing
    llama2_config = LLMConfig(case_name=case_name, data_parallel=2,
                              model_parallel=1,
                              pipeline_stage=2, micro_batch_num=2,
                              pp_interleave_num=2, pipeline_interleave=True, pipeline_scheduler="1f1b",
                              batch_size=1,
                              context_parallel=4,
                              parallel_speed_up_json={
                                  'enable_grad_comm_opt': 'false',
                                  'enable_opt_shard_comm_opt': 'false'},
                              select_recompute=True)
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    check_pair = {"Training Over": 1}
    # 返回路径 list
    graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    # 返回 validate.ir 的 graph_name
    validate_ir_graph_name = find_graph_file_name(graph_path[0], 'validate')
    # 返回 step_parallel_end.ir 的 graph_name
    parallel_end_ir_graph_name = find_graph_file_name(graph_path[0],
                                                      'step_parallel_end')
    parm_dpmp_node_strategy_check_pairs = {'PrimFunc_Gather': {'': '((4, 1), (2, 1))',}}
    parm_recompute_graph_check_pairs = {'recompute: Bool(1)': '7'}
    parm_opt_shape_check_pairs = {'_model.layers.0.attention.wq.weight': '(512, 4096)',
                                  '_model.layers.0.attention.wk.weight': '(512, 4096)',
                                  '_model.layers.0.attention.wv.weight': '(512, 4096)',
                                  '_accu_grads.model.layers.0.attention.wq.weight': '(4096, 4096)',
                                  '_accu_grads.model.layers.0.attention.wk.weight': '(4096, 4096)',
                                  '_accu_grads.model.layers.0.attention.wv.weight': '(4096, 4096)',
                                  '_adam_m.model.layers.0.attention.wq.weight': '(512, 4096)',
                                  '_adam_m.model.layers.0.attention.wk.weight': '(512, 4096)',
                                  '_adam_m.model.layers.0.attention.wv.weight': '(512, 4096)',
                                  }
    # 验证权重，两组
    # parm1_dependency_list = ['CNode_874', 8, 0, '_MicroStepAllGather']
    # parm2_dependency_list = ['', 9, 0, '_MicroStepAllGather']
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        # 检查编译时间，是否大于15、20
        check_compile_time(log_path, 15)
        # 检查内存，是否与预期相等
        check_peak_memory(log_path, '5200')
    # 检查 dp、mp，PrimFunc_Gather 的策略是否符合预期
    check_node_strategy(graph_path[0], validate_ir_graph_name,
                        parm_dpmp_node_strategy_check_pairs)
    # 检查选择重计算，recompute: Bool(1) 的个数
    check_graph(graph_path[0], validate_ir_graph_name,
                parm_recompute_graph_check_pairs)
    # 检查优化器，优化器/权重是否被切分(step_parallel_end.ir)
    check_param_shape(graph_path[0], parallel_end_ir_graph_name, 100,
                      parm_opt_shape_check_pairs)
    # 检查优化器, _MirrorMicroStepOperator，(step_parallel_end)
    # check_node_dependency_backward_search(graph_path[0], parallel_end_ir_graph_name, 100,
    #                                       parm1_dependency_list)
    # 检查优化器, _MirrorMicroStepOperator，(step_parallel_end)
    # check_node_dependency_backward_search(graph_path[0], parallel_end_ir_graph_name, 100,
    #                                       parm2_dependency_list)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_cell_dp2mp4pp2_fgi():
    """
    Feature: test llama2 cell_dp2mp4pp2_fgi
    Description: test llama2 cell_dp2mp4pp2_fgi
    Expectation: st pass
    """
    case_name = "llama2_cell_dp2mp4pp2_fgi"
    rank_list = "0,8"
    llama2_config = LLMConfig(case_name=case_name, data_parallel=2, model_parallel=4, pipeline_stage=2,
                              recompute=False, batch_size=1, vocab_emb_dp=False,
                              fine_grain_interleave=2, micro_batch_num=4,
                              use_seq_parallel=True, enable_parallel_optimizer=False,
                              parallel_speed_up_json={'matmul_grad_comm_overlap': 'true'})
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    check_pair = {"Training Over": 1}
    # 返回路径 list
    graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    # 返回 validate.ir 的 graph_name
    validate_ir_graph_name = find_graph_file_name(graph_path[0], 'validate')
    # PrimFunc_Gather 的策略
    parm_dpmp_strategy_check_pairs = {'PrimFunc_Gather': {'': '((4, 1), (2, 1))',}}
    # micro_interleaved_depend_begin 个数
    parm_micro_interleaved_depend_begin_check_pairs = {
        'micro_interleaved_depend_begin: Bool(1)': '4'}
    # 校验 rmsnorm 后面是否插入 tp 域的 allgather
    parm_tp_groups_check_pairs = {
        'AllGather': {'group_rank_ids': '((0, 1, 2, 3))',}}
    # 明确 reducescatter 的位置（rmsnorm后面）
    # pram_rmsnorm_reducescatter_dependency_list = ['CNode_837', 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    #                                               'PrimFunc_RmsNorm']
    # 序列并行, ReduceScatter 应和 AllGather 个数相等
    parm_reducescatter_allgather_check_pairs = {'ReduceScatter': '48',
                                                'AllGather': '48'}
    # 反向掩盖（mp/cp场景都开启）控制边名字个数
    parm_parallel_speed_up_check_pairs = {'grad_overlap_matmul': '22',
                                          'matmul_grad_depend2: Bool(1)': '11',
                                          'matmul_grad_depend3: Bool(1)': '11'}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_compile_time(log_path, 15)
        check_peak_memory(log_path, '4100')
    # 检查 dp、mp，PrimFunc_Gather 的策略是否符合预期
    check_node_strategy(graph_path[0], validate_ir_graph_name,
                        parm_dpmp_strategy_check_pairs)
    # fine_grain_interleave, 正常执行（不成环）统计控制边个数符合预期(micro_interleaved_depend_begin个数）
    check_graph(graph_path[0], validate_ir_graph_name,
                parm_micro_interleaved_depend_begin_check_pairs)
    # 校验 rmsnorm 后面是否插入 tp 域的 allgather
    check_comm_op_groups(graph_path[0], validate_ir_graph_name,
                         parm_tp_groups_check_pairs)
    # check_node_dependency_backward_search(graph_path[0], validate_ir_graph_name, 700,
    #                                       pram_rmsnorm_reducescatter_dependency_list)
    # 序列并行, reducescatter 应和 allgather 个数相等
    check_graph(graph_path[0], validate_ir_graph_name,
                parm_reducescatter_allgather_check_pairs)
    check_graph(graph_path[0], validate_ir_graph_name,
                parm_parallel_speed_up_check_pairs)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_cell_dp2mp1pp2cp4_fgi_grad_accu_select_recompute():
    """
    Feature: test llama2 cell_dp2mp4pp2_fgi
    Description: test llama2 cell_dp2mp4pp2_fgi
    Expectation: st pass
    """
    case_name = "llama2_cell_dp2mp1pp2cp4_fgi_grad_accu_select_recompute"
    rank_list = "0,8"
    llama2_config = LLMConfig(case_name=case_name, data_parallel=2, model_parallel=1, pipeline_stage=2,
                              recompute=True, select_recompute=True, batch_size=1, vocab_emb_dp=False,
                              micro_batch_num=4, fine_grain_interleave=2,
                              enable_parallel_optimizer=False, context_parallel=4,
                              parallel_speed_up_json={'matmul_grad_comm_overlap': 'true'})
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    check_pair = {"Training Over": 1}
    # 返回路径 list
    graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    # 返回 validate.ir 的 graph_name
    validate_ir_graph_name = find_graph_file_name(graph_path[0], 'validate')
    # PrimFunc_Gather 的策略
    parm_dpmp_strategy_check_pairs = {'PrimFunc_Gather': {'': '((4, 1), (2, 1))',}}
    # recompute: Bool(1) 数量
    parm_recompute_graph_check_pairs = {'recompute: Bool(1)': '281'}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_compile_time(log_path, 15)
    # 检查 dp、mp，PrimFunc_Gather 的策略是否符合预期
    check_node_strategy(graph_path[0], validate_ir_graph_name,
                        parm_dpmp_strategy_check_pairs)
    # 检查选择重计算，recompute: Bool(1) 的个数
    check_graph(graph_path[0], validate_ir_graph_name,
                parm_recompute_graph_check_pairs)
    check_peak_memory(real_log_path[0], '8200')
    check_peak_memory(real_log_path[1], '9800')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
@pytest.mark.skip(reason="Scalar")
def test_llama2_cell_dp2mp2pp1opcp2_fgi_grad_accu():
    """
    Feature: test llama2 cell_dp2mp4pp2_fgi
    Description: test llama2 cell_dp2mp4pp2_fgi
    Expectation: st pass
    """
    case_name = "llama2_dp2mp2pp1opcp2_fgi_grad_accu"
    rank_list = "0,4"
    llama2_config = LLMConfig(case_name=case_name, data_parallel=2, model_parallel=2, pipeline_stage=1,
                              recompute=False, batch_size=1, vocab_emb_dp=False,
                              gradient_accumulation_steps=4, fine_grain_interleave=2,
                              context_parallel=2,
                              parallel_speed_up_json={
                                  'matmul_grad_comm_overlap': 'true',
                                  'enable_grad_comm_opt': 'true',
                                  'enable_opt_shard_comm_opt': 'true',
                                  'dataset_broadcast_opt_level': 3},)
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name} no_pp")
    check_pair = {"Training Over": 1}
    # 返回路径 list
    graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    # 返回 validate.ir 的 graph_name
    validate_ir_graph_name = find_graph_file_name(graph_path[0], 'validate')
    # 返回 step_parallel_end.ir 的 graph_name
    parallel_end_ir_graph_name = find_graph_file_name(graph_path[0],
                                                      'step_parallel_end')
    # PrimFunc_Gather 的策略
    parm_dpmp_strategy_check_pairs = {'PrimFunc_Gather': {'': '((4, 1), (2, 1))',}}
    # 控制边数量 micro_interleaved_depend_begin
    parm_micro_interleaved_depend_begin_check_pairs = {
        'micro_interleaved_depend_begin: Bool(1)': '6'}
    # virtualassignadd 数量
    parm_virtualassignadd_check_pairs = {'VirtualAssignAdd': '66'}
    # 反向掩盖控制边个数
    parm_parallel_speed_up_check_pairs = {'grad_overlap_matmul': '54', 'matmul_grad_depend1': '0',
                                          'matmul_grad_depend2: Bool(1)': '27',
                                          'matmul_grad_depend3: Bool(1)': '27'}
    parm_opt_shape_check_pairs = {'_model.layers.0.attention.wq.weight': '(512, 4096)',
                                  '_model.layers.0.attention.wk.weight': '(512, 4096)',
                                  '_model.layers.0.attention.wv.weight': '(512, 4096)',
                                  '_accu_grads.model.layers.0.attention.wq.weight': '(2048, 4096)',
                                  '_accu_grads.model.layers.0.attention.wk.weight': '(2048, 4096)',
                                  '_accu_grads.model.layers.0.attention.wv.weight': '(2048, 4096)',
                                  '_adam_m.model.layers.0.attention.wq.weight': '(512, 4096)',
                                  '_adam_m.model.layers.0.attention.wk.weight': '(512, 4096)',
                                  '_adam_m.model.layers.0.attention.wv.weight': '(512, 4096)',
                                  }
    # 查找 _MicroStepAllGather
    # parm1_dependency_list = ['CNode_248', 0, '_MicroStepAllGather']
    # parm2_dependency_list = ['CNode_248', 2, '_MicroStepAllGather']
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_compile_time(log_path, 15)
        check_peak_memory(log_path, '6200')
    # 校验 dp、mp 的 Gather 的切分策略
    check_node_strategy(graph_path[0], validate_ir_graph_name,
                        parm_dpmp_strategy_check_pairs)
    # 细粒度多副本 fine_grain_interleave, 正常执行（不成环）统计控制边个数符合预期
    check_graph(graph_path[0], validate_ir_graph_name,
                parm_micro_interleaved_depend_begin_check_pairs)
    # 梯度累加，检查 virtualassignadd 数量
    check_graph(graph_path[0], parallel_end_ir_graph_name,
                parm_virtualassignadd_check_pairs)
    # 并行加速控制，校验反向掩盖控制边个数
    check_graph(graph_path[0], validate_ir_graph_name, parm_parallel_speed_up_check_pairs)
    # 检查优化器，优化器/权重是否被切分(step_parallel_end.ir)
    check_param_shape(graph_path[0], parallel_end_ir_graph_name, 100,
                      parm_opt_shape_check_pairs)
    # 检查优化器, _MirrorMicroStepOperator，(step_parallel_end)
    # check_node_dependency_backward_search(graph_path[0], parallel_end_ir_graph_name, 200, parm1_dependency_list)
    # 检查优化器, _MirrorMicroStepOperator，(step_parallel_end
    # check_node_dependency_backward_search(graph_path[0], parallel_end_ir_graph_name, 200, parm2_dependency_list)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
@pytest.mark.skip(reason="Scalar")
def test_llama2_cell_dp2mp2pp2vpp4opcp2_1f1b_grad_accu():
    """
    Feature: test llama2 cell_dp2mp2pp2vpp4opcp2_1f1b_grad_accu
    Description: test llama2 cell_dp2mp2pp2vpp4opcp2_1f1b_grad_accu
    Expectation: st pass
    """
    case_name = "llama2_cell_dp2mp2pp2vpp4opcp2_1f1b_grad_accu"
    rank_list = "0,8"
    llama2_config = LLMConfig(case_name=case_name, data_parallel=2,
                              model_parallel=2,
                              pipeline_stage=2, micro_batch_num=2,
                              pp_interleave_num=4, pipeline_interleave=True, pipeline_scheduler="1f1b",
                              batch_size=1,
                              context_parallel=2,
                              vocab_emb_dp=False,
                              num_layers=8,
                              parallel_speed_up_json={
                                  'matmul_grad_comm_overlap': 'true',
                                  'enable_grad_comm_opt': 'true',
                                  'enable_opt_shard_comm_opt': 'true',
                                  'dataset_broadcast_opt_level': 3},
                              gradient_accumulation_steps=2,
                              recompute=False)
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")
    check_pair = {"Training Over": 1}
    # 返回路径 list
    graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    # 返回 validate.ir 的 graph_name
    validate_ir_graph_name = find_graph_file_name(graph_path[0], 'validate')
    # 返回 step_parallel_end.ir 的 graph_name
    parallel_end_ir_graph_name = find_graph_file_name(graph_path[0],
                                                      'step_parallel_end')

    # PrimFunc_Gather 的策略
    parm_dpmp_strategy_check_pairs = {'PrimFunc_Gather': {'': '((4, 1), (2, 1))',}}
    # virtualassignadd 数量
    parm_virtualassignadd_check_pairs = {'VirtualAssignAdd': '74'}
    # 反向掩盖控制边个数
    parm_parallel_speed_up_check_pairs = {'grad_overlap_matmul': '26',
                                          'matmul_grad_depend2: Bool(1)': '13',
                                          'matmul_grad_depend3: Bool(1)': '13'}
    parm_opt_shape_check_pairs = {'_model.layers.0.attention.wq.weight': '(512, 4096)',
                                  '_model.layers.0.attention.wk.weight': '(512, 4096)',
                                  '_model.layers.0.attention.wv.weight': '(512, 4096)',
                                  '_accu_grads.model.layers.0.attention.wq.weight': '(2048, 4096)',
                                  '_accu_grads.model.layers.0.attention.wk.weight': '(2048, 4096)',
                                  '_accu_grads.model.layers.0.attention.wv.weight': '(2048, 4096)',
                                  '_adam_m.model.layers.0.attention.wq.weight': '(512, 4096)',
                                  '_adam_m.model.layers.0.attention.wk.weight': '(512, 4096)',
                                  '_adam_m.model.layers.0.attention.wv.weight': '(512, 4096)',
                                  }
    # 判断 _MirrorMicroStepOperator 的位置，查找 _MicroStepAllGather
    # parm1_dependency_list = ['CNode_1250', 0, '_MicroStepAllGather']
    # parm2_dependency_list = ['CNode_1250', 2, '_MicroStepAllGather']
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)
        check_compile_time(log_path, 15)
        check_peak_memory(log_path, '6200')
    # 校验 dp、mp 的 Gather 的切分策略
    check_node_strategy(graph_path[0], validate_ir_graph_name,
                        parm_dpmp_strategy_check_pairs)
    # 梯度累加，检查 virtualassignadd 数量
    check_graph(graph_path[0], parallel_end_ir_graph_name,
                parm_virtualassignadd_check_pairs)
    # 并行加速控制，校验反向掩盖控制边个数
    check_graph(graph_path[0], validate_ir_graph_name, parm_parallel_speed_up_check_pairs)
    # 检查优化器，优化器/权重是否被切分(step_parallel_end.ir)
    check_param_shape(graph_path[0], parallel_end_ir_graph_name, 100,
                      parm_opt_shape_check_pairs)
    # 检查优化器, _MirrorMicroStepOperator，(step_parallel_end)
    # check_node_dependency_backward_search(graph_path[0], parallel_end_ir_graph_name, 100, parm1_dependency_list)
    # 检查优化器, _MirrorMicroStepOperator，(step_parallel_end)
    # check_node_dependency_backward_search(graph_path[0], parallel_end_ir_graph_name, 100, parm2_dependency_list)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
@pytest.mark.skip(reason="Scalar")
def test_llama2_dp2mp2pp2opcp2_fgi_grad_accu_select_recompute():
    """
    Feature: test llama2 dp2mp2pp2opcp2_fgi_grad_accu
    Description: test llama2 dp2mp2pp2opcp2_fgi_grad_accu
    Expectation: st pass
    """
    case_name = "llama2_dp2mp2pp2opcp2_fgi_grad_accu"
    rank_list = "0,8"
    llama2_config = LLMConfig(data_parallel=2,
                              model_parallel=2,
                              pipeline_stage=2, micro_batch_num=2,
                              batch_size=1,
                              context_parallel=2,
                              vocab_emb_dp=False,
                              fine_grain_interleave=2,
                              parallel_speed_up_json={
                                  'matmul_grad_comm_overlap': 'true',
                                  'enable_grad_comm_opt': 'true',
                                  'enable_opt_shard_comm_opt': 'true',
                                  'dataset_broadcast_opt_level': 3},
                              gradient_accumulation_steps=4,
                              select_recompute=True)
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list}  {file_path} {output_file} {case_name}")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_cell_dp2mp2pp2vpp4opcp2_1f1b():
    """
    Feature: test llama2 cell_dp2mp2pp2vpp4opcp2_1f1b
    Description: test llama2 cell_dp2mp2pp2vpp4opcp2_1f1b
    Expectation: st pass
    """
    case_name = "llama2_cell_dp2mp2pp2vpp4opcp2_1f1b"
    rank_list = "0,8"
    llama2_config = LLMConfig(case_name=case_name,
                              parallel_mode=2,
                              data_parallel=2,
                              model_parallel=2,
                              pipeline_stage=2, micro_batch_num=2,
                              pp_interleave_num=4, pipeline_interleave=True, pipeline_scheduler="1f1b",
                              batch_size=2,
                              context_parallel=2,
                              num_layers=8,
                              vocab_emb_dp=False,
                              parallel_speed_up9_json={
                                  'matmul_grad_comm_overlap': 'true',
                                  'enable_grad_comm_opt': 'true',
                                  'enable_opt_shard_comm_opt': 'true',
                                  'dataset_broadcast_opt_level': 3},
                              gradient_accumulation_steps=2)
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name} pp")

    real_graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    graph_path = real_graph_path[0]
    # stage 0
    validate_name = find_graph_file_name(graph_path, "validate")
    # "step_parallel_end_0164.ir"
    step_parallel_end_name_0 = find_graph_file_name(graph_path, "step_parallel_end")
    # op 权重切分
    param_opt_shape_check_pairs = {"_model.layers.0.attention.wq.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wk.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wv.weight": "(512, 4096)",
                                   "_accu_grads.model.layers.0.attention.wq.weight": "(2048, 4096)",
                                   "_accu_grads.model.layers.0.attention.wk.weight": "(2048, 4096)",
                                   "_accu_grads.model.layers.0.attention.wv.weight": "(2048, 4096)"}
    check_param_shape(graph_path, step_parallel_end_name_0, 100, param_opt_shape_check_pairs)
    # op load 前有 allgather
    # param1_dependency_list = ['CNode_1437', 0, '_MicroStepAllGather']
    # param2_dependency_list = ['CNode_1437', 2, '_MicroStepAllGather']
    # check_node_dependency_backward_search(graph_path, step_parallel_end_name_0, 200, param1_dependency_list)
    # check_node_dependency_backward_search(graph_path, step_parallel_end_name_0, 200, param2_dependency_list)
    # cp allgather 前 matmul
    # param1_dependency_list = ['AllGather(%133)', 0, 0, 'PrimFunc_MatMul']
    # param2_dependency_list = ['CNode_2855) = AllGather(%194)', 0, 0, 'PrimFunc_MatMul']
    # check_node_dependency_backward_search(graph_path, validate_name, 200, param1_dependency_list)
    # check_node_dependency_backward_search(graph_path, validate_name, 200, param2_dependency_list)
    # recompute
    attrs_check_pairs = {"Grad__VirtualAssignAdd": 37,
                         }
    check_graph(graph_path, validate_name, attrs_check_pairs)

    # dp、mp Gather 切分
    gather_strategy_check_pairs = {"PrimFunc_Gather": {"": "((4, 1), (2, 1))"}}
    check_node_strategy(graph_path, validate_name, gather_strategy_check_pairs)
    # 梯度累加，检查 virtualassignadd 数量
    parm_virtualassignadd_check_pairs = {'VirtualAssignAdd': '74'}
    check_graph(graph_path, step_parallel_end_name_0, parm_virtualassignadd_check_pairs)

    # stage 1
    step_parallel_end_name_1 = find_graph_file_name(real_graph_path[1], "step_parallel_end")
    ops_check_pairs_1 = {"VirtualAssignAdd": 76}
    check_graph(real_graph_path[1], step_parallel_end_name_1, ops_check_pairs_1)
    check_param_shape(real_graph_path[0], validate_name, 100, param_opt_shape_check_pairs)
    # dependency_list_1_0 = ["CNode_1455", 0, "_MicroStepAllGather"]
    # dependency_list_1_1 = ["CNode_1455", 3, "_MicroStepAllGather"]
    # check_node_dependency_backward_search(real_graph_path[1], step_parallel_end_name_1, 100, dependency_list_1_0)
    # check_node_dependency_backward_search(real_graph_path[1], step_parallel_end_name_1, 100, dependency_list_1_1)

    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    log_path = None
    for log_path in real_log_path:
        check_log(log_path, check_pair)
    check_peak_memory(log_path, "9250")
    check_compile_time(log_path, 15)


@pytest.mark.skip(reason="has bug")
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_dp2mp2pp2cp2_fgi_grad_accu():
    """
    Feature: test llama2 cell_dp2mp2pp2vpp4opcp2_1f1b
    Description: test llama2 cell_dp2mp2pp2vpp4opcp2_1f1b
    Expectation: st pass
    """
    case_name = "llama2_dp2mp2pp2cp2_fgi_grad_accu"
    rank_list = "0,8"
    llama2_config = LLMConfig(parallel_mode=2,
                              data_parallel=2,
                              model_parallel=2,
                              pipeline_stage=2, micro_batch_num=2,
                              batch_size=2,
                              context_parallel=2,
                              fine_grain_interleave=2,
                              vocab_emb_dp=False,
                              parallel_speed_up_json={
                                  'matmul_grad_comm_overlap': 'true',
                                  'enable_grad_comm_opt': 'true',
                                  'enable_opt_shard_comm_opt': 'true',
                                  'dataset_broadcast_opt_level': 3},
                              gradient_accumulation_steps=4)
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name}")
    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    for log_path in real_log_path:
        check_log(log_path, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_dp4mp4pp1op_recompute_2():
    """
    Feature: test llama2 dp4mp4pp1op_recompute_2
    Description: test llama2 dp4mp4pp1op_recompute_2
    Expectation: st pass
    """
    case_name = "llama2_dp4mp4pp1op_recompute_2"
    rank_list = "0"
    llama2_config = LLMConfig(case_name=case_name,
                              data_parallel=4,
                              model_parallel=4,
                              batch_size=2,
                              vocab_emb_dp=False,
                              optimizer_weight_shard_size=2,
                              parallel_speed_up_json={
                                  'enable_grad_comm_opt:': 'true',
                                  'enable_opt_shard_comm_opt:True': 'true'},
                              recompute=True)
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {rank_list} {file_path} {output_file} {case_name}")
    real_graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    graph_path = real_graph_path[0]

    validate_name = find_graph_file_name(graph_path, "validate")
    step_parallel_end_name = find_graph_file_name(graph_path, "step_parallel_end")
    # op 权重切分
    param_opt_shape_check_pairs = {"_model.layers.0.attention.wq.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wk.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wv.weight": "(512, 4096)"}
    check_param_shape(graph_path, step_parallel_end_name, 100, param_opt_shape_check_pairs)
    # op load 前有 allgather
    # param1_dependency_list = ["CNode_96", 0, "AllGather"]
    # param2_dependency_list = ["CNode_101", 0, "AllGather"]
    # check_node_dependency_backward_search(graph_path, step_parallel_end_name, 100, param1_dependency_list)
    # check_node_dependency_backward_search(graph_path, step_parallel_end_name, 100, param2_dependency_list)
    # op 不切满场景 反向有ReduceScatter + AllReduce
    # dependency_list = ['AllReduce(%860)', 0, 'ReduceScatter(%859)']
    # check_node_dependency_backward_search(graph_path, validate_name, 100, dependency_list)

    # recompute
    attrs_check_pairs = {' recompute: Bool(1)': '206'}
    check_graph(graph_path, validate_name, attrs_check_pairs)

    # dp Gather 切分
    gather_strategy_check_pairs = {"PrimFunc_Gather": {"": "((4, 1), (4, 1))"}}
    check_node_strategy(graph_path, validate_name, gather_strategy_check_pairs)

    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    log_path = None
    for log_path in real_log_path:
        check_log(log_path, check_pair)
    check_peak_memory(log_path, "4100")
    check_compile_time(log_path, 15)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='dryrun_only', essential_mark='essential')
def test_llama2_dp8mp1pp1op():
    """
    Feature: test llama2 dp8mp1pp1op
    Description: test llama2 dp8mp1pp1op
    Expectation: st pass
    """
    case_name = "llama2_dp8mp1pp1op"
    rank_list = "0"
    llama2_config = LLMConfig(case_name=case_name,
                              data_parallel=8,
                              model_parallel=1,
                              batch_size=2)
    output_file, file_path = prepare_testcase_env(case_name, llama2_config)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {rank_list} {file_path} {output_file} {case_name}")

    real_graph_path = graph_path_preprocess(llama2_config.save_graphs_path, rank_list)
    graph_path = real_graph_path[0]
    validate_name = find_graph_file_name(graph_path, "validate")
    step_parallel_end_name = find_graph_file_name(graph_path, "step_parallel_end")
    # op 权重切分
    param_opt_shape_check_pairs = {"_model.layers.0.attention.wq.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wk.weight": "(512, 4096)",
                                   "_model.layers.0.attention.wv.weight": "(512, 4096)"}
    check_param_shape(graph_path, step_parallel_end_name, 100, param_opt_shape_check_pairs)
    # op load 前有 allgather

    # param1_dependency_list = ["CNode_96", 0, "AllGather"]
    # param2_dependency_list = ["CNode_125", 0, "AllGather"]
    # check_node_dependency_backward_search(graph_path, step_parallel_end_name, 100, param1_dependency_list)
    # check_node_dependency_backward_search(graph_path, step_parallel_end_name, 100, param2_dependency_list)
    # # dp Gather 切分
    gather_strategy_check_pairs = {"PrimFunc_Gather": {"output": "((1, 1), (8, 1))"}}
    check_node_strategy(graph_path, validate_name, gather_strategy_check_pairs)

    check_pair = {"Training Over": 1}
    real_log_path = log_path_preprocess(output_file, rank_list, case_name)
    log_path = None
    for log_path in real_log_path:
        check_log(log_path, check_pair)
    check_peak_memory(log_path, "14350")
    check_compile_time(log_path, 15)
