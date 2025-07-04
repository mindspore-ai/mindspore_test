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
Test module for testing silent check.
How to run this:
pytest tests/st/silent_check/test_silent_check.py
"""
import os
import subprocess
import pytest
from tests.mark_utils import arg_mark

def exec_command(cmd):
    s = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    out = s.stdout.read().decode("UTF-8")
    s.stdout.close()
    return out


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_npu_asd_enable0():
    """
    Feature: Test silent check with NPU_ASD_ENABLE=0
    Description: Test silent check with NPU_ASD_ENABLE=0.
    Expectation: mindspore graph does not contain SilentCheckV2 operator
    """
    os.environ['NPU_ASD_ENABLE'] = "0"
    os.environ['MS_SAVE_GRAPHS'] = "1"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret1 = os.system(f"bash {sh_path}/msrun_silent_check.sh --master_port=8150 {sh_path}/silent_check.py")
    ret2 = os.system(f"ls ms_graphs/rank_0/*.ir | grep silent_check")
    ret3 = os.system(f"grep -E 'SilentCheckV2' ms_graphs/rank_0/graph_build*.ir")
    assert ret1 == 0
    assert ret2 != 0
    assert ret3 != 0
    os.system(f'rm -rf ms_graphs worker_*.log ascend_log')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_npu_asd_enable1():
    """
    Feature: Test silent check with NPU_ASD_ENABLE=1
    Description: Test silent check with NPU_ASD_ENABLE=1.
    Expectation: training exit normally but with SilentCheck ERROR in ascend log
    """
    os.environ['NPU_ASD_ENABLE'] = "1"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret1 = os.system(f"bash {sh_path}/msrun_silent_check.sh --master_port=8151 {sh_path}/silent_check.py")
    ret2 = os.system(f"grep -E -nr -m1 'ERROR.*silent_check_v[23].cc.*SilentCheck' ascend_log/")
    assert ret1 == 0
    assert ret2 == 0
    os.system(f'rm -rf ms_graphs worker_*.log ascend_log')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_npu_asd_enable2():
    """
    Feature: Test silent check with NPU_ASD_ENABLE=2
    Description: Test silent check with NPU_ASD_ENABLE=2.
    Expectation: training exit abnormally but with ONLY SilentCheck ERROR in ascend log
    """
    os.environ['NPU_ASD_ENABLE'] = "2"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret1 = os.system(f"bash {sh_path}/msrun_silent_check.sh --master_port=8152 {sh_path}/silent_check.py &> /dev/null")
    ret2 = os.system(f"grep -E -nr -m1 'ERROR.*silent_check_v[23].cc.*SilentCheck' ascend_log/")
    ret3 = os.system(f"grep -E -nr -m1 'INFO.*silent_check_v[23].cc.*SilentCheck' ascend_log/")
    assert ret1 != 0
    assert ret2 == 0
    assert ret3 != 0
    os.system(f'rm -rf ms_graphs worker_*.log ascend_log')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_npu_asd_enable3():
    """
    Feature: Test silent check with NPU_ASD_ENABLE=3
    Description: Test silent check with NPU_ASD_ENABLE=3.
    Expectation: training exit abnormally but with BOTH SilentCheck INFO and ERROR in ascend log
    """
    os.environ['NPU_ASD_ENABLE'] = "3"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret1 = os.system(f"bash {sh_path}/msrun_silent_check.sh --master_port=8153 {sh_path}/silent_check.py &> /dev/null")
    ret2 = os.system(f"grep -E -nr -m1 'ERROR.*silent_check_v[23].cc.*SilentCheck' ascend_log/")
    ret3 = os.system(f"grep -E -nr -m1 'INFO.*silent_check_v[23].cc.*SilentCheck' ascend_log/")
    assert ret1 != 0
    assert ret2 == 0
    assert ret3 == 0
    os.system(f'rm -rf ms_graphs worker_*.log ascend_log')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
def test_silent_check_receive():
    """
    Feature: Test silent check not insert check operator for receive operator.
    Description: Test silent check not insert check operator for receive operator.
    Expectation: SilentCheckV2 operator was not inserted for receive operator.
    """
    os.environ['NPU_ASD_ENABLE'] = "3"
    os.environ['MS_SAVE_GRAPHS'] = "1"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    cmd = f"bash {sh_path}/msrun_silent_check.sh --master_port=8154 {sh_path}/pipeline_parallel.py &> /dev/null"
    ret1 = os.system(cmd)
    ret2 = os.system(f"ls ms_graphs/rank_0/*.ir | grep silent_check")
    ret3 = os.system(f"grep -E -nr -m1 'INFO.*silent_check_v[23].cc.*SilentCheck' ascend_log/")
    assert ret1 == 0
    assert ret2 == 0
    assert ret3 == 0

    os.system(f'rm -rf ms_graphs worker_*.log ascend_log')


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='allcards', essential_mark='unessential')
@pytest.mark.parametrize("fp16_type", ("fp16_weight", "fp16_input", "fp16_getnext"))
def test_silent_check_skip_float16_inputs(fp16_type):
    """
    Feature: Test silent check not insert check operator when network has fp16 weight.
    Description: Test silent check not insert check operator when network has fp16 weight.
    Expectation: SilentCheckV2 operator was not inserted to graph.
    """
    ports_map = {"fp16_weight": 8155, "fp16_input": 8156, "fp16_getnext": 8157}
    os.environ['NPU_ASD_ENABLE'] = "3"
    os.environ['MS_SAVE_GRAPHS'] = "1"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    cmd = f"bash {sh_path}/msrun_silent_check.sh --master_port={ports_map[fp16_type]}" \
          f" {sh_path}/data_parallel.py {fp16_type}&> /dev/null"
    ret1 = os.system(cmd)
    ret2 = os.system(f"cat ms_graphs/rank_0/graph_build*.ir | grep '= PrimFunc_SilentCheckV2('")
    ret3 = os.system(f"cat worker_0.log | grep ', skip inserting silent check operators'")
    assert ret1 == 0
    assert ret2 != 0
    assert ret3 == 0
