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
import pytest


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_npu_asd_enable0():
    """
    Feature: Test silent check with NPU_ASD_ENABLE=0
    Description: Test silent check with NPU_ASD_ENABLE=0.
    Expectation: mindspore graph does not contain SilentCheckV2 operator
    """
    os.environ['NPU_ASD_ENABLE'] = "0"
    os.environ['MS_SAVE_GRAPHS'] = "1"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret1 = os.system(f"bash {sh_path}/mpirun_silent_check.sh")
    ret2 = os.system(f"ls ms_graphs/rank_0/verbose_ir_files/*.ir | grep insert_silent_check_v2")
    ret3 = os.system(f"grep -E 'SilentCheckV2' ms_graphs/rank_0/graph_build*.ir")
    assert ret1 == 0
    assert ret2 != 0
    assert ret3 != 0
    os.system(f'rm -rf ms_graphs log_output ascend_log')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_npu_asd_enable1():
    """
    Feature: Test silent check with NPU_ASD_ENABLE=1
    Description: Test silent check with NPU_ASD_ENABLE=1.
    Expectation: training exit normally but with SilentCheck ERROR in ascend log
    """
    os.environ['NPU_ASD_ENABLE'] = "1"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret1 = os.system(f"bash {sh_path}/mpirun_silent_check.sh")
    ret2 = os.system(f"grep -E -nr -m1 'ERROR.*silent_check_v2.cc.*SilentCheck get L' ascend_log/")
    assert ret1 == 0
    assert ret2 == 0
    os.system(f'rm -rf ms_graphs log_output ascend_log')


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_npu_asd_enable2():
    """
    Feature: Test silent check with NPU_ASD_ENABLE=2
    Description: Test silent check with NPU_ASD_ENABLE=2.
    Expectation: training exit abnormally but with ONLY SilentCheck ERROR in ascend log
    """
    os.environ['NPU_ASD_ENABLE'] = "2"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret1 = os.system(f"bash {sh_path}/mpirun_silent_check.sh &> /dev/null")
    ret2 = os.system(f"grep -E -nr -m1 'ERROR.*silent_check_v2.cc.*SilentCheck get L' ascend_log/")
    ret3 = os.system(f"grep -E -nr -m1 'INFO.*silent_check_v2.cc.*SilentCheck' ascend_log/")
    assert ret1 != 0
    assert ret2 == 0
    assert ret3 != 0
    os.system(f'rm -rf ms_graphs log_output ascend_log')


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_npu_asd_enable3():
    """
    Feature: Test silent check with NPU_ASD_ENABLE=3
    Description: Test silent check with NPU_ASD_ENABLE=3.
    Expectation: training exit abnormally but with BOTH SilentCheck INFO and ERROR in ascend log
    """
    os.environ['NPU_ASD_ENABLE'] = "3"
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    ret1 = os.system(f"bash {sh_path}/mpirun_silent_check.sh &> /dev/null")
    ret2 = os.system(f"grep -E -nr -m1 'ERROR.*silent_check_v2.cc.*SilentCheck get L' ascend_log/")
    ret3 = os.system(f"grep -E -nr -m1 'INFO.*silent_check_v2.cc.*SilentCheck' ascend_log/")
    assert ret1 != 0
    assert ret2 == 0
    assert ret3 == 0
    os.system(f'rm -rf ms_graphs log_output ascend_log')
