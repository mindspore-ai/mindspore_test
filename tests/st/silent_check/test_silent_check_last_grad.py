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
"""
import os
import pytest


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_silent_check1():
    """
    Feature: Test silent check for last grad node
    Description: Test silent check in non-sink mode
    Expectation: SilentCheckV2 operator is executed
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    py_file = 'silent_check_last_grad1.py'
    ret1 = os.system(f"bash {sh_path}/singlerun_silent_check.sh {sh_path}/{py_file}")
    ret2 = os.system(f"grep -E -nr -m1 'INFO.*silent_check_v2.cc.*SilentCheck' ascend_log/")
    assert ret1 == 0
    assert ret2 == 0
    os.system(f'rm -rf ms_graphs log_output ascend_log')


@pytest.mark.level0
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_single
def test_silent_check2():
    """
    Feature: Test silent check for last grad node
    Description: Test silent check in sink mode
    Expectation: SilentCheckV2 operator is executed
    """
    sh_path = os.path.split(os.path.realpath(__file__))[0]
    py_file = 'silent_check_last_grad2.py'
    ret1 = os.system(f"bash {sh_path}/singlerun_silent_check.sh {sh_path}/{py_file}")
    ret2 = os.system(f"grep -E -nr -m1 'INFO.*silent_check_v2.cc.*SilentCheck' ascend_log/")
    assert ret1 == 0
    assert ret2 == 0
    os.system(f'rm -rf ms_graphs log_output ascend_log')
