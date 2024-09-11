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
Test module for parallel training of Llama models using Mindformers at jit_level O2.
"""
import os
import shutil
import subprocess
from tests.mark_utils import arg_mark


def run_command(cmd, log_path, graph_path, graph_check, log_check, loss_check):
    if os.path.exists(graph_path):
        shutil.rmtree(graph_path)
    if os.path.isfile(log_path):
        os.remove(log_path)
    os.system(cmd)

    graph_file = f"{graph_path}/hwopt_d_after_stream_assign*.ir"
    graph_para = "GEGraphOp("
    graph_output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (graph_para, graph_file)],
        shell=True)
    graph_cnt = str(graph_output, 'utf-8').strip()
    assert graph_cnt == str(graph_check)

    log_para = "Start compile"
    log_output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (log_para, log_path)],
        shell=True)
    log_cnt = str(log_output, 'utf-8').strip()
    assert log_cnt == str(log_check)

    loss_para = "loss:"
    loss_output = subprocess.check_output(
        ["grep -r '%s' %s | wc -l" % (loss_para, log_path)],
        shell=True)
    loss_cnt = str(loss_output, 'utf-8').strip()
    assert loss_cnt == str(loss_check)

    print(graph_cnt, log_cnt, loss_cnt, flush=True)

    if os.path.exists(graph_path):
        shutil.rmtree(graph_path)
    if os.path.isfile(log_path):
        os.remove(log_path)


def run_command_compile(cmd, log_path, backend_time, compile_time):
    if os.path.isfile(log_path):
        os.remove(log_path)
    os.system(cmd)

    log_backend = "compile_backend_graph costs"
    log_output = subprocess.check_output(
        ["grep -r '%s' %s | awk '{print $3}'" % (log_backend, log_path)],
        shell=True)
    log_time = str(log_output, 'utf-8').strip()
    assert float(log_time) <= backend_time

    log_compile = "compile_graph costs"
    log_output = subprocess.check_output(
        ["grep -r '%s' %s | awk '{print $3}'" % (log_compile, log_path)],
        shell=True)
    log_time = str(log_output, 'utf-8').strip()
    assert float(log_time) <= compile_time

    if os.path.isfile(log_path):
        os.remove(log_path)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_train_compile():
    """
    Feature: Trainer.train()
    Description: Test llama2 70b compile time.
    Expectation: AssertionError
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command_compile(f"bash {sh_path}/dry_compile.sh compile", f"{sh_path}/compile.log", 80000, 220000)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_train_pipeline():
    """
    Feature: Trainer.train()
    Description: Test context parallel trainer for train.
    Expectation: AssertionError
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/dry.sh 0 pipeline", f"{sh_path}/pipeline.log",
                f"{sh_path}/pipeline/rank_0", 4, 2, 10)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_train_grad_accu():
    """
    Feature: Trainer.train()
    Description: Test context parallel trainer for train.
    Expectation: AssertionError
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    run_command(f"bash {sh_path}/dry.sh 1 grad_accu", f"{sh_path}/grad_accu.log",
                f"{sh_path}/grad_accu/rank_0", 8, 2, 10)
