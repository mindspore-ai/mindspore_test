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

# 1. copy yaml from mindformers
# 2. replace parts of value in yaml through replace_config func
# 3. run st in dryrun modeyfg

import os
import pytest
import subprocess

from tests.mark_utils import arg_mark


def replace_config(net_config, file_path):
    old_list = [
        'dataset_dir: ""', 'enable_parallel_optimizer: True', 'vocab_emb_dp: True',
        'full_batch: True', 'num_layers: 32', 'gradient_accumulation_steps: 8', 'batch_size: 1',
        'batch_size: 6', 'micro_batch_num: 1', 'data_parallel: 8', 'model_parallel: 1', 'pipeline_stage: 1',
        'epochs: 2', 'sink_size: 2', 'recompute: False', 'select_recompute: False', 'use_seq_parallel: False',
        'offset: 0', "output_dir: './output'", 'save_graphs: False', 'save_graphs_path: "./graph"'
    ]

    new_list = [
        f'dataset_dir: {net_config.dataset_dir}', f'enable_parallel_optimizer: {net_config.enable_parallel_optimizer}',
        f'vocab_emb_dp: {net_config.vocab_emb_dp}', f'full_batch: {net_config.full_batch}',
        f'num_layers: {net_config.num_layers}',
        f'gradient_accumulation_steps: {net_config.gradient_accumulation_steps}',
        f'batch_size: {net_config.batch_size}', f'batch_size: {net_config.batch_size}',
        f'micro_batch_num: {net_config.micro_batch_num}', f'data_parallel: {net_config.data_parallel}',
        f'model_parallel: {net_config.model_parallel}', f'pipeline_stage: {net_config.pipeline_stage}',
        f'epochs: {net_config.epochs}', f'sink_size: {net_config.sink_size}', f'recompute: {net_config.recompute}',
        f'select_recompute: {net_config.select_recompute}', f'use_seq_parallel: {net_config.use_seq_parallel}',
        f'offset: {net_config.offset}', f"output_dir: '{net_config.output_dir}'",
        f'save_graphs: {net_config.save_graphs}', f'save_graphs_path: "{net_config.save_graphs_path}"'
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
    # remove checkpoint monitor to prevent saving ckpt
    remove_checkpoint_monitor = f"sed -i '/CheckpointMonitor/,+4d' {file_path}"
    status, _ = subprocess.getstatusoutput(remove_checkpoint_monitor)
    if status != 0:
        print(f"Failed to remove CheckpointMonitor in {file_path}")
        return False
    # insert fine grain interleaved
    if net_config.fine_grain_interleave > 1:
        insert_fine_grain_interleave = r"sed -i '/model_config:/a\    fine_grain_interleave: {}' {}".format(
            net_config.fine_grain_interleave, file_path
        )
        status, _ = subprocess.getstatusoutput(insert_fine_grain_interleave)
        if status != 0:
            print(f"Failed to insert fine grain interleave in {file_path}")
    # insert context parallel
    if net_config.context_parallel:
        insert_context_parallel = r"sed -i '/vocab_emb_dp:/i\  context_parallel: {}' {}".format(
            net_config.context_parallel, file_path
        )
        status, _ = subprocess.getstatusoutput(insert_context_parallel)
        if status != 0:
            print(f"Failed to insert context parallel in {file_path}")
    # insert pipeline interleaved
    if net_config.pipeline_interleave:
        if net_config.pipeline_scheduler is None or net_config.pp_interleave_num == -1:
            print("pipeline_scheduler and pp_interleave_num should be set together")
            return False
        insert_pipeline_config = r"sed -i '/full_batch:/i\  pipeline_config:' {}".format(file_path)
        insert_pipeline_interleave = r"sed -i '/full_batch:/i\    pipeline_interleave: {}' {}".format(
            net_config.pipeline_interleave, file_path)
        insert_pipeline_scheduler = r"""sed -i '/full_batch:/i\    pipeline_scheduler: "{}"' {}""".format(
            net_config.pipeline_scheduler, file_path)
        insert_pp_interleave_num = r"""sed -i '/model_config:/a\    pp_interleave_num: {}' {}""".format(
            net_config.pp_interleave_num, file_path)
        for cmd in [insert_pipeline_config, insert_pipeline_interleave, insert_pipeline_scheduler,
                    insert_pp_interleave_num]:
            status, _ = subprocess.getstatusoutput(cmd)
            if status != 0:
                print(f"Failed to execute cmd {cmd} in {file_path}")
    return True


def check_log(file_path, check_pairs=None):
    log_error_count = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % ("ERROR", file_path)],
        shell=True)
    log_cnt = str(log_error_count, 'utf-8').strip()
    assert log_cnt == "0", f"Error found in {file_path}"
    if check_pairs is not None:
        for key_word, value in check_pairs.items():
            log_output = subprocess.check_output(
                ["grep -r '%s' %s | wc -l" % (key_word, file_path)],
                shell=True)
            log_cnt = str(log_output, 'utf-8').strip()
            assert log_cnt == str(value), f"Failed to find {key_word} in {file_path} or content is not correct"


class Llama2Config:
    # add default config for llama2
    def __init__(self,
                 dataset_dir="/home/workspace/mindspore_dataset/wiki4096/wiki4096.mindrecord",
                 enable_parallel_optimizer=True,
                 vocab_emb_dp=True,
                 full_batch=True,
                 num_layers=4,
                 gradient_accumulation_steps=1,
                 batch_size=1,
                 micro_batch_num=1,
                 data_parallel=8,
                 model_parallel=1,
                 pipeline_stage=1,
                 epochs=2,
                 sink_size=2,
                 recompute=False,
                 select_recompute=False,
                 use_seq_parallel=False,
                 offset=0,
                 output_dir="./output",
                 save_graphs=False,
                 save_graphs_path="./graph",
                 num_samples=64,
                 fine_grain_interleave=1,
                 context_parallel=False,
                 pipeline_interleave=False,
                 pipeline_scheduler=None,
                 pp_interleave_num=-1,
                 **kwargs):
        # output dir
        self.output_dir = output_dir

        # context
        self.save_graphs = save_graphs
        self.save_graphs_path = save_graphs_path

        # dataset
        self.dataset_dir = dataset_dir
        self.num_samples = num_samples

        # parallel context
        self.enable_parallel_optimizer = enable_parallel_optimizer
        self.full_batch = full_batch
        self.pipeline_interleave = pipeline_interleave
        self.pipeline_scheduler = pipeline_scheduler

        # parallel
        self.data_parallel = data_parallel
        self.model_parallel = model_parallel
        self.pipeline_stage = pipeline_stage
        self.context_parallel = context_parallel
        self.vocab_emb_dp = vocab_emb_dp
        self.micro_batch_num = micro_batch_num
        self.use_seq_parallel = use_seq_parallel

        # model config
        self.num_layers = num_layers
        self.fine_grain_interleave = fine_grain_interleave
        self.offset = offset
        self.pp_interleave_num = pp_interleave_num

        # runner config
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.epochs = epochs
        self.sink_size = sink_size

        # recompute
        self.recompute = recompute
        self.select_recompute = select_recompute


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_llama2_dp2mp4pp1_recompute():
    """
    Feature: test llama2 dp2mp4pp1 full_recompute
    Description: test llama2 dp2mp4pp1 full_recompute
    Expectation: st pass
    """
    output_file = "dp2mp4pp1_recompute_output.log"
    llama2_config = Llama2Config(data_parallel=2, model_parallel=4,
                                 enable_parallel_optimizer=False, batch_size=4, vocab_emb_dp=False)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    file_path = f'{sh_path}/llama2_config/dp2mp4pp1_recompute.yaml'
    os.system("cp '../mindformers/configs/llama2/pretrain_llama2_7b_bf16.yaml' '{}'".format(file_path))
    status = replace_config(llama2_config, file_path)
    if not status:
        raise Exception("Failed to replace config in {}".format(file_path))
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {file_path} {output_file}")
    check_pair = {"Training Over": 1}
    check_log(output_file, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_llama2_dp4mp4pp1op_recompute():
    """
    Feature: test llama2 dp4mp4pp1op full_recompute
    Description: test llama2 dp4mp4pp1op full_recompute
    Expectation: st pass
    """
    output_file = "dp4mp4pp1op_recompute_output.log"
    llama2_config = Llama2Config(data_parallel=4, model_parallel=4,
                                 recompute=True, batch_size=2, vocab_emb_dp=False)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    file_path = f'{sh_path}/llama2_config/dp4mp4pp1op_recompute.yaml'
    os.system("cp '../mindformers/configs/llama2/pretrain_llama2_7b_bf16.yaml' '{}'".format(file_path))
    status = replace_config(llama2_config, file_path)
    if not status:
        raise Exception("Failed to replace config in {}".format(file_path))
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {file_path} {output_file}")
    check_pair = {"Training Over": 1}
    check_log(output_file, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_llama2_cell_dp2mp4pp1op_grad_accu():
    """
    Feature: test llama2 cell_dp2mp4pp1op_grad_accu
    Description: test llama2 cell_dp2mp4pp1op_grad_accu
    Expectation: st pass
    """
    output_file = "cell_dp2mp4pp1_grad_accu_output.log"
    llama2_config = Llama2Config(data_parallel=2, model_parallel=4,
                                 gradient_accumulation_steps=4, batch_size=1,
                                 recompute=False)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    file_path = f'{sh_path}/llama2_config/cell_dp2mp4pp1op_grad_accu.yaml'
    os.system("cp '../mindformers/configs/llama2/pretrain_llama2_7b_bf16.yaml' '{}'".format(file_path))
    status = replace_config(llama2_config, file_path)
    if not status:
        raise Exception("Failed to replace config in {}".format(file_path))
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 8 {file_path} {output_file} no_pp")
    check_pair = {"Training Over": 1}
    check_log(output_file, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_llama2_cell_dp2mp4pp2vpp4_1f1b():
    """
    Feature: test llama2 cell_dp2mp4pp1op_grad_accu
    Description: test llama2 cell_dp2mp4pp1op_grad_accu
    Expectation: st pass
    """
    output_file = "cell_dp2mp4pp2vpp4_1f1b_output.log"
    llama2_config = Llama2Config(data_parallel=2, model_parallel=4, pipeline_stage=2,
                                 micro_batch_num=2, batch_size=2, pp_interleave_num=4,
                                 pipeline_interleave=True, pipeline_scheduler="1f1b",
                                 num_layers=8, recompute=False)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    file_path = f'{sh_path}/llama2_config/cell_dp2mp4pp2vpp4_1f1b.yaml'
    os.system("cp '../mindformers/configs/llama2/pretrain_llama2_7b_bf16.yaml' '{}'".format(file_path))
    status = replace_config(llama2_config, file_path)
    if not status:
        raise Exception("Failed to replace config in {}".format(file_path))
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {file_path} {output_file} pp")
    check_pair = {"Training Over": 1}
    check_log(output_file, check_pair)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_llama2_cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute():
    """
    Feature: test llama2 cell_dp2mp4pp1op_grad_accu
    Description: test llama2 cell_dp2mp4pp1op_grad_accu
    Expectation: st pass
    """
    output_file = "cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute_output.log"
    llama2_config = Llama2Config(data_parallel=2, model_parallel=1, pipeline_stage=2,
                                 micro_batch_num=4, batch_size=1, pp_interleave_num=2,
                                 pipeline_interleave=True, pipeline_scheduler="1f1b",
                                 num_layers=4, context_parallel=4, select_recompute=True,
                                 recompute=True)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    file_path = f'{sh_path}/llama2_config/cell_dp2mp1pp2vpp2cp4_1f1b_select_recompute.yaml'
    os.system("cp '../mindformers/configs/llama2/pretrain_llama2_7b_bf16.yaml' '{}'".format(file_path))
    status = replace_config(llama2_config, file_path)
    if not status:
        raise Exception("Failed to replace config in {}".format(file_path))
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {file_path} {output_file} pp")
    check_pair = {"Training Over": 1}
    check_log(output_file, check_pair)


@pytest.mark.skip(reason="In current version of fgi, there is a graph cycle issue, so we skip this case")
@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_llama2_cell_dp2mp4pp2_fgi():
    """
    Feature: test llama2 cell_dp2mp4pp2_fgi
    Description: test llama2 cell_dp2mp4pp2_fgi
    Expectation: st pass
    """
    output_file = "cell_dp2mp4pp2_fgi_output.log"
    llama2_config = Llama2Config(data_parallel=2, model_parallel=4, pipeline_stage=2,
                                 recompute=False, batch_size=1, vocab_emb_dp=False,
                                 fine_grain_interleave=2, micro_batch_num=4,
                                 use_seq_parallel=True, enable_parallel_optimizer=False)
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    file_path = f'{sh_path}/llama2_config/cell_dp2mp4pp2_fgi.yaml'
    os.system("cp '../mindformers/configs/llama2/pretrain_llama2_7b_bf16.yaml' '{}'".format(file_path))
    status = replace_config(llama2_config, file_path)
    if not status:
        raise Exception("Failed to replace config in {}".format(file_path))
    os.system(f"bash {sh_path}/run_llm_dryrun.sh 16 {file_path} {output_file} pp")
    check_pair = {"Training Over": 1}
    check_log(output_file, check_pair)
